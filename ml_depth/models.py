import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, DINOv2Model
from peft import LoraConfig, get_peft_model


class LoRACrossAttention(nn.Module):
    """
    Cross-attention module with LoRA adaptation
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, lora_r=16, lora_alpha=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, key, value projections with LoRA
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        # Apply LoRA to the projections
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.1,
        )
        
        # The PEFT library expects a module with specific structure, so we need to adapt
        # our projections to be wrapped with LoRA
        self.q_proj_lora = get_peft_model(nn.Sequential(self.q_proj), lora_config)
        self.k_proj_lora = get_peft_model(nn.Sequential(self.k_proj), lora_config)
        self.v_proj_lora = get_peft_model(nn.Sequential(self.v_proj), lora_config)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value):
        """
        Cross-attention forward pass
        Args:
            query: tensor of shape [B, L_q, D]
            key: tensor of shape [B, L_k, D]
            value: tensor of shape [B, L_v, D]
        Returns:
            tensor of shape [B, L_q, D]
        """
        B, L_q, D = query.size()
        _, L_k, _ = key.size()
        
        # Apply LoRA projections
        q = self.q_proj_lora(query)[:, 0]  # Get the first element since the Sequential adds a dimension
        k = self.k_proj_lora(key)[:, 0]
        v = self.v_proj_lora(value)[:, 0]
        
        # Reshape for multi-head attention
        q = q.view(B, L_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, L_q, head_dim]
        k = k.view(B, L_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, L_k, head_dim]
        v = v.view(B, L_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, L_v, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, num_heads, L_q, L_k]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = torch.matmul(attn_probs, v)  # [B, num_heads, L_q, head_dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L_q, D)  # [B, L_q, D]
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class StereoTransformerNet(nn.Module):
    def __init__(self, 
                 dinov2_model_name="facebook/dinov2-small", 
                 lora_r=16, 
                 lora_alpha=32, 
                 d_model=384,  # DINOv2-small embedding dimension
                 nhead=8,
                 num_cross_attn_layers=2):
        """
        Stereo Transformer using DINOv2 models with LoRA adapters
        
        Args:
            dinov2_model_name: Name or path of the DINOv2 model
            lora_r: LoRA rank parameter
            lora_alpha: LoRA alpha parameter
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_cross_attn_layers: Number of cross-attention layers
        """
        super(StereoTransformerNet, self).__init__()
        
        # Initialize image processor for DINOv2
        self.image_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        
        # Initialize two DINOv2 models - one for left image, one for right image
        self.dinov2_left = DINOv2Model.from_pretrained(dinov2_model_name)
        self.dinov2_right = DINOv2Model.from_pretrained(dinov2_model_name)
        
        # Apply LoRA to the DINOv2 models
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.1,
        )
        
        self.dinov2_left = get_peft_model(self.dinov2_left, lora_config)
        self.dinov2_right = get_peft_model(self.dinov2_right, lora_config)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            LoRACrossAttention(dim=d_model, num_heads=nhead, lora_r=lora_r, lora_alpha=lora_alpha)
            for _ in range(num_cross_attn_layers)
        ])
        
        # Self-attention layers
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=0.1)
            for _ in range(2)  # Two self-attention layers as requested
        ])
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        
        # Linear layer
        self.linear = nn.Linear(d_model, d_model)
        
        # Disparity head for regression
        self.disparity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Upsampling factor (DINOv2 uses patch size of 14 by default)
        self.patch_size = 14
        
    def forward(self, left, right):
        """
        Forward pass
        
        Args:
            left: Left stereo image [B, 3, H, W]
            right: Right stereo image [B, 3, H, W]
            
        Returns:
            disparity_map: Disparity map [B, 1, H, W]
        """
        B, _, H, W = left.shape
        
        # Preprocess images for DINOv2 (if needed)
        # Note: If images are already normalized according to DINOv2 requirements,
        # this step can be skipped
        
        # Get DINOv2 embeddings
        left_embeddings = self.dinov2_left(left).last_hidden_state  # [B, L, D]
        right_embeddings = self.dinov2_right(right).last_hidden_state  # [B, L, D]
        
        # Apply cross-attention between left and right embeddings
        attn_output = left_embeddings
        for cross_attn in self.cross_attn_layers:
            attn_output = self.norm1(attn_output + cross_attn(attn_output, right_embeddings, right_embeddings))
        
        # Apply self-attention
        for self_attn in self.self_attn_layers:
            # Self-attention expects input of shape [L, B, D]
            attn_input = attn_output.transpose(0, 1)
            attn_out, _ = self_attn(attn_input, attn_input, attn_input)
            attn_output = self.norm2(attn_output + attn_out.transpose(0, 1))
        
        # Apply linear layer
        linear_output = self.linear(attn_output)
        linear_output = F.relu(linear_output)
        linear_output = self.norm_out(linear_output)
        
        # Apply disparity head
        disparity_tokens = self.disparity_head(linear_output)  # [B, L, 1]
        
        # Reshape to 2D disparity map
        # The number of tokens L = (H // patch_size) * (W // patch_size) + 1 (CLS token)
        # We exclude the CLS token and reshape the remaining tokens
        tokens_without_cls = disparity_tokens[:, 1:, 0]  # [B, L-1]
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        
        # Reshape to [B, 1, h_patches, w_patches]
        disparity_map = tokens_without_cls.view(B, h_patches, w_patches, 1).permute(0, 3, 1, 2)
        
        # Upsample to original resolution
        disparity_map = F.interpolate(
            disparity_map, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Ensure positive disparities (apply softplus or similar activation)
        disparity_map = F.softplus(disparity_map)
        
        return disparity_map