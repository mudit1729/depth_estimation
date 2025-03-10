import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedDisparityLoss(nn.Module):
    def __init__(self, weights={
        'berhu': 1.0, 
        'scale_invariant': 0.5, 
        'edge_smooth': 0.1, 
        'normal_smooth': 0.05
    }, berhu_threshold=0.2):
        """
        Combined disparity loss with robust regression, scale invariance, and edge-aware smoothness
        
        Args:
            weights (dict): Weighting for different loss components
            berhu_threshold (float): Threshold for BerHu loss
        """
        super(CombinedDisparityLoss, self).__init__()
        self.weights = weights
        self.berhu_threshold = berhu_threshold
        
    def forward(self, pred_disp, gt_disp, left_img=None):
        """
        Combined loss function for disparity prediction
        
        Args:
            pred_disp (torch.Tensor): Predicted disparity [B, 1, H, W]
            gt_disp (torch.Tensor): Ground truth disparity [B, 1, H, W]
            left_img (torch.Tensor, optional): Left image for edge-aware smoothness [B, 3, H, W]
        """
        # Valid mask - only consider pixels with ground truth
        valid_mask = (gt_disp > 0).float()
        num_valid = torch.sum(valid_mask) + 1e-8
        
        # Initialize total loss
        total_loss = 0.0
        loss_details = {}
        
        # 1. BerHu (reverse Huber) loss - robust to outliers
        if self.weights.get('berhu', 0) > 0:
            berhu_loss = self.berhu_loss(pred_disp, gt_disp, valid_mask, self.berhu_threshold)
            total_loss += self.weights['berhu'] * berhu_loss
            loss_details['berhu'] = berhu_loss.item()
        
        # 2. Scale-invariant loss for global scale consistency
        if self.weights.get('scale_invariant', 0) > 0:
            si_loss = self.scale_invariant_loss(pred_disp, gt_disp, valid_mask)
            total_loss += self.weights['scale_invariant'] * si_loss
            loss_details['scale_invariant'] = si_loss.item()
        
        # 3. Edge-aware smoothness loss
        if left_img is not None and self.weights.get('edge_smooth', 0) > 0:
            edge_smooth_loss = self.edge_aware_smoothness_loss(pred_disp, left_img)
            total_loss += self.weights['edge_smooth'] * edge_smooth_loss
            loss_details['edge_smooth'] = edge_smooth_loss.item()
        
        # 4. Normal smoothness loss (encourages planar surfaces)
        if self.weights.get('normal_smooth', 0) > 0:
            normal_smooth_loss = self.normal_smoothness_loss(pred_disp)
            total_loss += self.weights['normal_smooth'] * normal_smooth_loss
            loss_details['normal_smooth'] = normal_smooth_loss.item()
            
        return total_loss, loss_details
    
    def berhu_loss(self, pred, target, mask, threshold=0.2):
        """
        BerHu loss (reverse Huber) - L1 for small residuals, L2 for large residuals
        
        Args:
            pred (torch.Tensor): Predicted values
            target (torch.Tensor): Target values
            mask (torch.Tensor): Valid pixel mask
            threshold (float): Threshold for switching between L1 and L2
        """
        diff = torch.abs(target - pred) * mask
        c = threshold * torch.max(diff).detach()
        
        # L1 loss for small residuals
        l1_mask = (diff <= c).float() * mask
        l1_loss = diff * l1_mask
        
        # L2 loss for large residuals
        l2_mask = (diff > c).float() * mask
        l2_loss = (diff * diff + c * c) / (2 * c) * l2_mask
        
        # Combine losses
        loss = l1_loss + l2_loss
        return torch.sum(loss) / (torch.sum(mask) + 1e-8)
    
    def scale_invariant_loss(self, pred, target, mask):
        """
        Scale-invariant loss - invariant to global scaling factors
        
        Args:
            pred (torch.Tensor): Predicted values
            target (torch.Tensor): Target values
            mask (torch.Tensor): Valid pixel mask
        """
        # Convert to log space
        pred_log = torch.log(pred + 1e-8) * mask
        target_log = torch.log(target + 1e-8) * mask
        
        # Difference in log space
        log_diff = (pred_log - target_log) * mask
        
        # Number of valid pixels
        num_valid = torch.sum(mask) + 1e-8
        
        # Scale-invariant loss: mean of squared log difference minus variance term
        si_loss = torch.sum(log_diff ** 2) / num_valid - 0.5 * (torch.sum(log_diff) / num_valid) ** 2
        
        return si_loss
    
    def edge_aware_smoothness_loss(self, disp, img):
        """
        Enhanced edge-aware smoothness loss for disparity
        
        Args:
            disp (torch.Tensor): Disparity map [B, 1, H, W]
            img (torch.Tensor): Reference image [B, 3, H, W]
        """
        # Normalize disparity for scale-invariant gradients
        mean_disp = torch.mean(disp, dim=(2, 3), keepdim=True)
        disp_normalized = disp / (mean_disp + 1e-8)
        
        # Get disparity gradients (both first and second order)
        disp_grad_x = torch.abs(disp_normalized[:, :, :, :-1] - disp_normalized[:, :, :, 1:])
        disp_grad_y = torch.abs(disp_normalized[:, :, :-1, :] - disp_normalized[:, :, 1:, :])
        
        # Get image gradients
        img_grad_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
        img_grad_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)
        
        # Apply weighting based on image gradients (stronger edges = less smoothness constraint)
        # Use exponential to get sharper edge response
        weight_x = torch.exp(-img_grad_x * 10.0)
        weight_y = torch.exp(-img_grad_y * 10.0)
        
        # Weighted smoothness - penalize disparity changes where image is smooth
        smoothness_x = disp_grad_x * weight_x
        smoothness_y = disp_grad_y * weight_y
        
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)
    
    def normal_smoothness_loss(self, disp):
        """
        Normal vector smoothness loss - encourages planar surfaces
        
        Args:
            disp (torch.Tensor): Disparity map [B, 1, H, W]
        """
        # Get disparity gradients
        disp_dx = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        disp_dy = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        
        # Second-order gradients (encourage planar surfaces)
        disp_dxx = torch.abs(disp_dx[:, :, :, :-1] - disp_dx[:, :, :, 1:])
        disp_dxy = torch.abs(disp_dx[:, :, :-1, :] - disp_dx[:, :, 1:, :])
        disp_dyx = torch.abs(disp_dy[:, :, :, :-1] - disp_dy[:, :, :, 1:])
        disp_dyy = torch.abs(disp_dy[:, :, :-1, :] - disp_dy[:, :, 1:, :])
        
        # Combine all second-order terms
        return torch.mean(disp_dxx) + torch.mean(disp_dxy) + torch.mean(disp_dyx) + torch.mean(disp_dyy)


# Legacy/Simplified loss functions for backward compatibility
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