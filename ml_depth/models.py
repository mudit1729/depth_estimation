import torch
import torch.nn as nn
import torch.nn.functional as F

class StereoTransformerNet(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        """
        d_model: Transformer embedding dimension (should match cost volume channels)
        nhead: Number of attention heads
        num_layers: Number of Transformer encoder layers
        """
        super(StereoTransformerNet, self).__init__()
        
        # CNN feature extractor for each image
        # Downsamples input by a factor of 8 and outputs 128-channel features.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, H/8, W/8)
            nn.ReLU(inplace=True),
        )
        # After feature extraction, each image yields a (B, 128, H/8, W/8) feature map.
        # We concatenate left and right features to obtain a cost volume of shape (B, 256, H/8, W/8).
        
        # Transformer encoder to fuse spatial context from the cost volume.
        # The cost volume is reshaped into a sequence where each token corresponds to one spatial location.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression head: maps each transformer token to a disparity value.
        self.disparity_head = nn.Linear(d_model, 1)
        
    def forward(self, left, right):
        # left, right: (B, 3, H, W)
        left_feat = self.feature_extractor(left)   # -> (B, 128, H/8, W/8)
        right_feat = self.feature_extractor(right)   # -> (B, 128, H/8, W/8)
        
        # Concatenate along the channel dimension -> (B, 256, H/8, W/8)
        cost_volume = torch.cat([left_feat, right_feat], dim=1)
        B, C, H, W = cost_volume.shape
        
        # Flatten the spatial dimensions to form a sequence:
        # (B, C, H, W) -> (B, H*W, C)
        cost_seq = cost_volume.view(B, C, H * W).permute(0, 2, 1)  # shape: (B, S, d_model) with S = H*W
        
        # Transformer expects input of shape (S, B, d_model)
        cost_seq = cost_seq.permute(1, 0, 2)  # shape: (S, B, d_model)
        transformer_out = self.transformer(cost_seq)  # shape: (S, B, d_model)
        
        # Map each token (spatial location) to a disparity value
        disparity_seq = self.disparity_head(transformer_out)  # shape: (S, B, 1)
        
        # Reshape back to a 2D disparity map:
        disparity_seq = disparity_seq.permute(1, 2, 0)  # shape: (B, 1, S)
        disparity_map = disparity_seq.view(B, 1, H, W)    # shape: (B, 1, H/8, W/8)
        
        # Upsample disparity map to match the original image resolution if needed.
        disparity_map = F.interpolate(disparity_map, scale_factor=8, mode='bilinear', align_corners=False)
        
        # Note: To obtain depth, use the relation:
        #       depth = (focal_length * baseline) / disparity_map
        return disparity_map