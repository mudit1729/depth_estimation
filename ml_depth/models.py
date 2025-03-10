import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoImageProcessor, Dinov2Model
from peft import LoraConfig, get_peft_model


class DeformableAttention(nn.Module):
    """
    Deformable Attention module for stereo feature fusion.
    This implementation is a simplified version of the deformable attention
    mechanism described in the "Deformable DETR" paper. It allows the attention
    module to focus on specific spatial locations by learning offsets to sampling points.
    """
    def __init__(self, dim, num_heads=8, n_points=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.n_points = n_points
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections for query, key, value
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        
        # Offset prediction network
        # For each attention head and each sampling point, predict a 2D offset
        self.offset_proj = nn.Linear(dim, num_heads * n_points * 2)
        
        # Sampling weight prediction network
        # These weights determine the importance of each sampling point
        self.weight_proj = nn.Linear(dim, num_heads * n_points)
        
        self.dropout = nn.Dropout(dropout)
        
    def _get_deformable_attention(self, q, reference_points, input_spatial_shapes, offset, weight, v):
        """
        Apply deformable attention with learned offsets and weights.
        
        Args:
            q: Query features [B, nH, Lq, d]
            reference_points: Reference points in normalized [0, 1] space [B, nH, Lq, n_points, 2]
            input_spatial_shapes: Spatial shape of the input [h, w]
            offset: Predicted offsets for reference points [B, nH, Lq, n_points, 2]
            weight: Weights for sampling points [B, nH, Lq, n_points]
            v: Value features [B, nH, Lv, d]
            
        Returns:
            output: Output after applying deformable attention [B, nH, Lq, d]
        """
        B, nH, Lq, d = q.shape
        _, _, Lv, _ = v.shape
        h, w = input_spatial_shapes
        
        # Add offsets to the reference points
        # Normalize offsets to [-1, 1] range
        sampling_locations = reference_points + offset
        sampling_locations = torch.clamp(sampling_locations, 0, 1)
        
        # Scale to feature map coordinates
        sampling_locations_h = sampling_locations[:, :, :, :, 0] * (h - 1)
        sampling_locations_w = sampling_locations[:, :, :, :, 1] * (w - 1)
        
        # Get grid coordinates for grid_sample
        # Rescale to [-1, 1] for grid_sample
        sampling_grids_h = 2.0 * sampling_locations_h / (h - 1) - 1.0  # [B, nH, Lq, n_points]
        sampling_grids_w = 2.0 * sampling_locations_w / (w - 1) - 1.0  # [B, nH, Lq, n_points]
        
        # Reshape value for attention computation
        v_reshape = v.reshape(B, nH, h * w, d)  # [B, nH, Lv, d]
        
        # Initialize output tensor
        output = torch.zeros_like(q)  # [B, nH, Lq, d]
        
        # For each head and each query location, sample from the value map
        for i in range(B):
            for j in range(nH):
                for p in range(self.n_points):
                    # Get grid coordinates for this sampling point
                    grid_h = sampling_grids_h[i, j, :, p]  # [Lq]
                    grid_w = sampling_grids_w[i, j, :, p]  # [Lq]
                    
                    # Create the sampling grid for grid_sample
                    # Expected shape for grid_sample: [Lq, 1, 2]
                    grid = torch.stack([grid_w, grid_h], dim=-1).unsqueeze(1)
                    
                    # Reshape value features for this head
                    # Expected shape for grid_sample input: [1, d, h, w]
                    v_head = v_reshape[i, j].transpose(0, 1).reshape(d, h, w).unsqueeze(0)
                    
                    # Sample from the value map using bilinear interpolation
                    sampled_feat = F.grid_sample(
                        v_head, grid, mode='bilinear', padding_mode='zeros', align_corners=True
                    )  # [1, d, Lq, 1]
                    
                    # Reshape the sampled features and apply weights
                    sampled_feat = sampled_feat.squeeze(-1).squeeze(0).transpose(0, 1)  # [Lq, d]
                    output[i, j] += weight[i, j, :, p].unsqueeze(-1) * sampled_feat
        
        return output
    
    def forward(self, query, key, value):
        """
        Deformable attention forward pass.
        
        Args:
            query: Tensor of shape [B, Lq, D]
            key: Tensor of shape [B, Lk, D]
            value: Tensor of shape [B, Lv, D]
            
        Returns:
            output: Tensor of shape [B, Lq, D]
        """
        B, Lq, D = query.shape
        _, Lk, _ = key.shape
        _, Lv, _ = value.shape
        
        # Compute token spatial shapes (assuming square feature maps for simplicity)
        # In a real implementation, you would derive this from the input dimensions
        feature_size = int(math.sqrt(Lk))
        input_spatial_shapes = torch.tensor([feature_size, feature_size])
        
        # Project query, key, value
        q = self.q_proj(query)  # [B, Lq, D]
        k = self.k_proj(key)    # [B, Lk, D]
        v = self.v_proj(value)  # [B, Lv, D]
        
        # Generate reference points for the deformable attention
        # We'll use a grid of reference points across the feature map
        h, w = input_spatial_shapes
        
        # Create a grid of reference points in [0, 1] for the query
        # For simplicity, we'll use the same grid for all queries
        y_ref = torch.linspace(0, 1, h).reshape(-1, 1).repeat(1, w).reshape(-1)
        x_ref = torch.linspace(0, 1, w).reshape(1, -1).repeat(h, 1).reshape(-1)
        ref_2d = torch.stack([x_ref, y_ref], dim=-1)  # [Lk, 2]
        
        # Repeat reference points for each batch and head
        ref_2d = ref_2d.unsqueeze(0).unsqueeze(0).repeat(B, self.num_heads, 1, 1, 1)  # [B, nH, Lk, 1, 2]
        ref_2d = ref_2d.repeat(1, 1, 1, self.n_points, 1)  # [B, nH, Lk, n_points, 2]
        
        # Project to sampling locations (reference points + offsets)
        # Predict offsets for sampling locations
        offset = self.offset_proj(query)  # [B, Lq, nH * n_points * 2]
        offset = offset.reshape(B, Lq, self.num_heads, self.n_points, 2)  # [B, Lq, nH, n_points, 2]
        offset = offset.permute(0, 2, 1, 3, 4)  # [B, nH, Lq, n_points, 2]
        
        # Predict weights for sampling points
        weight = self.weight_proj(query)  # [B, Lq, nH * n_points]
        weight = weight.reshape(B, Lq, self.num_heads, self.n_points)  # [B, Lq, nH, n_points]
        weight = weight.permute(0, 2, 1, 3)  # [B, nH, Lq, n_points]
        weight = F.softmax(weight, dim=-1)  # Normalize weights within each head
        
        # Reshape for multi-head attention
        q = q.reshape(B, Lq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nH, Lq, d]
        k = k.reshape(B, Lk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nH, Lk, d]
        v = v.reshape(B, Lv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nH, Lv, d]
        
        # Apply deformable attention
        attn_output = self._get_deformable_attention(q, ref_2d, input_spatial_shapes, offset, weight, v)
        
        # Reshape output
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, Lq, D)
        
        # Output projection
        attn_output = self.output_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        return attn_output


class LoRACrossAttention(nn.Module):
    """
    Cross-attention module for stereo feature fusion.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, lora_r=16, lora_alpha=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Using PyTorch's built-in MultiheadAttention
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value):
        """
        Cross-attention forward pass.
        Args:
            query: tensor of shape [B, L_q, D]
            key: tensor of shape [B, L_k, D]
            value: tensor of shape [B, L_v, D]
        Returns:
            tensor of shape [B, L_q, D]
        """
        attn_output, _ = self.mha(query, key, value)
        attn_output = self.dropout(attn_output)
        return attn_output


class ResidualBlock(nn.Module):
    """
    A basic residual block with two 3x3 convolutions, batch normalization, and ReLU activation.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResNetRefinementModule(nn.Module):
    """
    A ResNet-style refinement module that uses residual blocks to predict a disparity offset.
    It concatenates the left image and the coarse disparity map, processes them through an
    initial convolution, several residual blocks, and a final convolution layer.
    """
    def __init__(self, in_channels=4, base_channels=64, num_blocks=4):
        """
        Args:
            in_channels: Number of input channels (3 for left image + 1 for coarse disparity).
            base_channels: Number of channels for the convolutional layers.
            num_blocks: Number of residual blocks.
        """
        super(ResNetRefinementModule, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.initial_bn = nn.BatchNorm2d(base_channels)
        self.initial_relu = nn.ReLU(inplace=True)
        
        # Stack of residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_blocks)]
        )
        
        # Final convolution to predict the residual offset
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, image, coarse_disp):
        # image: [B, 3, H, W], coarse_disp: [B, 1, H, W]
        x = torch.cat([image, coarse_disp], dim=1)  # [B, 4, H, W]
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = self.res_blocks(x)
        disp_offset = self.final_conv(x)
        refined_disp = coarse_disp + disp_offset
        refined_disp = F.softplus(refined_disp)  # Ensure positive disparities
        return refined_disp


class StereoTransformerNet(nn.Module):
    def __init__(self, 
                 dinov2_model_name="facebook/dinov2-small", 
                 lora_r=16, 
                 lora_alpha=32, 
                 d_model=384,  # DINOv2-small embedding dimension
                 nhead=8,
                 num_cross_attn_layers=2):
        """
        Stereo Transformer using DINOv2 models with LoRA adapters and a ResNet-style refinement module.
        
        Args:
            dinov2_model_name: Name or path of the DINOv2 model.
            lora_r: LoRA rank parameter.
            lora_alpha: LoRA alpha parameter.
            d_model: Embedding dimension.
            nhead: Number of attention heads.
            num_cross_attn_layers: Number of cross-attention layers.
        """
        super(StereoTransformerNet, self).__init__()
        
        # Initialize image processor for DINOv2
        self.image_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        
        # Initialize two DINOv2 models - one for the left image, one for the right image
        self.dinov2_left = Dinov2Model.from_pretrained(dinov2_model_name)
        self.dinov2_right = Dinov2Model.from_pretrained(dinov2_model_name)
        
        # Apply LoRA to the DINOv2 models
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value", "out_proj", "fc1", "fc2"],
            lora_dropout=0.1,
        )
        self.dinov2_left = get_peft_model(self.dinov2_left, lora_config)
        self.dinov2_right = get_peft_model(self.dinov2_right, lora_config)
        
        # Cross-attention layers with deformable attention 
        self.cross_attn_layers = nn.ModuleList([
            DeformableAttention(dim=d_model, num_heads=nhead, n_points=4, dropout=0.1)
            for _ in range(num_cross_attn_layers)
        ])
        
        # Self-attention layers also using deformable attention
        # This helps focus on the most relevant spatial locations within each view
        self.self_attn_layers = nn.ModuleList([
            DeformableAttention(dim=d_model, num_heads=nhead, n_points=4, dropout=0.1)
            for _ in range(2)  # Two self-attention layers
        ])
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        
        # Linear layer before disparity head
        self.linear = nn.Linear(d_model, d_model)
        
        # Disparity head for regression
        self.disparity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Upsampling factor (DINOv2 uses a patch size of 14 by default)
        self.patch_size = 14

        # ResNet-style refinement module to further enhance the disparity map
        self.refinement_module = ResNetRefinementModule(in_channels=4, base_channels=64, num_blocks=4)
        
    def forward(self, left, right):
        """
        Forward pass.
        
        Args:
            left: Left stereo image [B, 3, H, W].
            right: Right stereo image [B, 3, H, W].
            
        Returns:
            refined_disparity_map: Refined disparity map [B, 1, H, W].
        """
        B, _, H, W = left.shape
        
        # Process images through DINOv2 models
        left_outputs = self.dinov2_left(left, output_hidden_states=True)
        right_outputs = self.dinov2_right(right, output_hidden_states=True)
        
        # Get the last hidden state from each model
        left_embeddings = left_outputs.last_hidden_state  # [B, L, D]
        right_embeddings = right_outputs.last_hidden_state  # [B, L, D]
        
        # Apply deformable cross-attention between left and right embeddings
        # This allows the model to focus on specific regions in the right image
        # based on features in the left image
        attn_output = left_embeddings
        for cross_attn in self.cross_attn_layers:
            attn_output = self.norm1(attn_output + cross_attn(attn_output, right_embeddings, right_embeddings))
        
        # Apply deformable self-attention layers
        # This helps refine features within the cross-attended output
        for self_attn in self.self_attn_layers:
            attn_out = self_attn(attn_output, attn_output, attn_output)
            attn_output = self.norm2(attn_output + attn_out)
        
        # Apply linear layer with ReLU activation
        linear_output = F.relu(self.linear(attn_output))
        linear_output = self.norm_out(linear_output)
        
        # Compute coarse disparity tokens using the disparity head
        disparity_tokens = self.disparity_head(linear_output)  # [B, L, 1]
        
        # Calculate the expected number of patches
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        
        # Reshape tokens into a 2D disparity map
        if disparity_tokens.shape[1] == h_patches * w_patches + 1:  # if a CLS token is present
            tokens_without_cls = disparity_tokens[:, 1:, 0]  # [B, L-1]
            disparity_patches = tokens_without_cls.view(B, h_patches, w_patches)
        else:
            disparity_patches = disparity_tokens[:, :, 0].view(B, h_patches, w_patches)
        
        # Add channel dimension: [B, h_patches, w_patches] -> [B, 1, h_patches, w_patches]
        disparity_patches = disparity_patches.unsqueeze(1)
        
        # Upsample to the original resolution
        coarse_disparity_map = F.interpolate(
            disparity_patches, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Ensure positive disparities
        coarse_disparity_map = F.softplus(coarse_disparity_map)
        
        # Refine the disparity map using the ResNet-style refinement module
        refined_disparity_map = self.refinement_module(left, coarse_disparity_map)
        
        return refined_disparity_map
