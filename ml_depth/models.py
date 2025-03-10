import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoImageProcessor, Dinov2Model
from peft import LoraConfig, get_peft_model


class SimpleMultiheadAttention(nn.Module):
    """
    A simpler implementation of multihead attention to replace the deformable attention
    that was causing compatibility issues.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        """
        Multi-head attention forward pass.
        
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
        
        # Linear projections
        q = self.q_proj(query)  # [B, Lq, D]
        k = self.k_proj(key)    # [B, Lk, D]
        v = self.v_proj(value)  # [B, Lv, D]
        
        # Reshape for multi-head attention
        q = q.reshape(B, Lq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, Lq, d]
        k = k.reshape(B, Lk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, Lk, d]
        v = v.reshape(B, Lv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, Lv, d]
        
        # Compute attention weights
        # [B, H, Lq, d] × [B, H, d, Lk] -> [B, H, Lq, Lk]
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        # [B, H, Lq, Lk] × [B, H, Lv, d] -> [B, H, Lq, d]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, Lq, D)
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
                 num_cross_attn_layers=4,  # Increased from 2 to 4
                 num_self_attn_layers=4):  # Added explicit parameter
        """
        Stereo Transformer using DINOv2 models with LoRA adapters and a ResNet-style refinement module.
        
        Args:
            dinov2_model_name: Name or path of the DINOv2 model.
            lora_r: LoRA rank parameter.
            lora_alpha: LoRA alpha parameter.
            d_model: Embedding dimension.
            nhead: Number of attention heads.
            num_cross_attn_layers: Number of cross-attention layers (default: 4).
            num_self_attn_layers: Number of self-attention layers (default: 4).
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
        
        # Cross-attention layers using simple multihead attention
        self.cross_attn_layers = nn.ModuleList([
            SimpleMultiheadAttention(dim=d_model, num_heads=nhead, dropout=0.1)
            for _ in range(num_cross_attn_layers)  # Increased to 4 cross-attention layers
        ])
        
        # Self-attention layers using simple multihead attention
        self.self_attn_layers = nn.ModuleList([
            SimpleMultiheadAttention(dim=d_model, num_heads=nhead, dropout=0.1)
            for _ in range(num_self_attn_layers)  # Increased to 4 self-attention layers
        ])
        
        # Layer normalization layers for each attention layer
        # Cross-attention layer norms
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) 
            for _ in range(num_cross_attn_layers)
        ])
        
        # Self-attention layer norms
        self.self_attn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) 
            for _ in range(num_self_attn_layers)
        ])
        
        # Final layer norm
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
        
        # Apply cross-attention between left and right embeddings
        # This allows the model to find correspondences between the left and right views
        attn_output = left_embeddings
        for i, cross_attn in enumerate(self.cross_attn_layers):
            # Apply cross-attention and add residual connection
            cross_attn_output = cross_attn(attn_output, right_embeddings, right_embeddings)
            attn_output = self.cross_attn_norms[i](attn_output + cross_attn_output)
        
        # Apply self-attention layers to refine the features
        for i, self_attn in enumerate(self.self_attn_layers):
            # Apply self-attention and add residual connection
            self_attn_output = self_attn(attn_output, attn_output, attn_output)
            attn_output = self.self_attn_norms[i](attn_output + self_attn_output)
        
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
