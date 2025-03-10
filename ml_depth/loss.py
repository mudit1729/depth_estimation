import torch
import torch.nn as nn
import torch.nn.functional as F

class DisparityLoss(nn.Module):
    def __init__(self, weights={'l1': 1.0, 'smooth': 0.1}):
        """
        Disparity Loss - combination of L1 loss and smoothness regularization
        
        Args:
            weights (dict): Weighting for different loss components
        """
        super(DisparityLoss, self).__init__()
        self.weights = weights
        
    def forward(self, pred_disp, gt_disp, left_img=None):
        """
        Args:
            pred_disp (torch.Tensor): Predicted disparity [B, 1, H, W]
            gt_disp (torch.Tensor): Ground truth disparity [B, 1, H, W]
            left_img (torch.Tensor, optional): Left image for edge-aware smoothness [B, 3, H, W]
        """
        # Create valid mask - only consider pixels where ground truth is available
        valid_mask = (gt_disp > 0).float()
        
        # L1 loss - only for valid pixels
        l1_loss = torch.mean(torch.abs(pred_disp - gt_disp) * valid_mask) / (torch.mean(valid_mask) + 1e-7)
        
        loss = self.weights['l1'] * l1_loss
        
        # Add edge-aware smoothness term if left image is provided
        if left_img is not None and self.weights.get('smooth', 0) > 0:
            smooth_loss = self.edge_aware_smoothness_loss(pred_disp, left_img)
            loss += self.weights['smooth'] * smooth_loss
            
        return loss
    
    def edge_aware_smoothness_loss(self, disp, img):
        """
        Edge-aware smoothness loss for disparity
        
        Args:
            disp (torch.Tensor): Disparity map [B, 1, H, W]
            img (torch.Tensor): Reference image [B, 3, H, W]
        """
        # Get image gradients
        img_grad_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
        img_grad_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)
        
        # Get disparity gradients
        disp_grad_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        disp_grad_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        
        # Apply weighting based on image gradients (edges)
        weight_x = torch.exp(-img_grad_x)
        weight_y = torch.exp(-img_grad_y)
        
        smoothness_x = disp_grad_x * weight_x
        smoothness_y = disp_grad_y * weight_y
        
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)


class SmoothL1DisparityLoss(nn.Module):
    """
    SmoothL1 (Huber) loss for disparity estimation
    """
    def __init__(self, beta=1.0):
        super(SmoothL1DisparityLoss, self).__init__()
        self.beta = beta
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=beta)
    
    def forward(self, pred_disp, gt_disp):
        """
        Args:
            pred_disp (torch.Tensor): Predicted disparity [B, 1, H, W]
            gt_disp (torch.Tensor): Ground truth disparity [B, 1, H, W]
        """
        # Create valid mask
        valid_mask = (gt_disp > 0).float()
        
        # Apply SmoothL1 loss
        loss = self.smooth_l1(pred_disp, gt_disp)
        
        # Apply mask and normalize
        loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-7)
        
        return loss