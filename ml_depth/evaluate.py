import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# Import local modules
from models import StereoTransformerNet
from dataset import KITTIDataset, ToTensor, Normalize, Resize

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a stereo depth estimation model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to KITTI dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--resize', type=int, nargs=2, default=[256, 512], help='Resize input images (height, width)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to evaluate')
    
    return parser.parse_args()

def calculate_metrics(pred_disp, gt_disp, focal, baseline):
    """
    Calculate common disparity and depth estimation metrics
    
    Args:
        pred_disp: Predicted disparity (B, 1, H, W)
        gt_disp: Ground truth disparity (B, 1, H, W)
        focal: Focal length
        baseline: Baseline
    
    Returns:
        metrics: Dictionary of metrics
    """
    # Create mask for valid pixels
    valid_mask = (gt_disp > 0).float()
    
    # Calculate per-pixel absolute disparity error
    abs_diff = torch.abs(pred_disp - gt_disp) * valid_mask
    
    # Calculate error metrics for disparity
    metrics = {
        'abs_rel_disp': (abs_diff / (gt_disp + 1e-10) * valid_mask).sum() / (valid_mask.sum() + 1e-10),
        'rmse_disp': torch.sqrt((abs_diff ** 2).sum() / (valid_mask.sum() + 1e-10)),
        'mae_disp': abs_diff.sum() / (valid_mask.sum() + 1e-10),
        'a1_disp': (torch.max(gt_disp / (pred_disp + 1e-10), pred_disp / (gt_disp + 1e-10)) < 1.25).float().sum() / (valid_mask.sum() + 1e-10),
    }
    
    # Convert to depth
    pred_depth = (focal * baseline) / (pred_disp + 1e-10)
    gt_depth = (focal * baseline) / (gt_disp + 1e-10)
    
    # Handle invalid depths (infinite or very large values)
    depth_mask = (valid_mask * (gt_depth < 80) * (pred_depth < 80)).float()
    
    # Calculate per-pixel absolute depth error
    abs_diff_depth = torch.abs(pred_depth - gt_depth) * depth_mask
    
    # Calculate error metrics for depth
    metrics.update({
        'abs_rel_depth': (abs_diff_depth / (gt_depth + 1e-10) * depth_mask).sum() / (depth_mask.sum() + 1e-10),
        'rmse_depth': torch.sqrt(((pred_depth - gt_depth)**2 * depth_mask).sum() / (depth_mask.sum() + 1e-10)),
        'mae_depth': abs_diff_depth.sum() / (depth_mask.sum() + 1e-10),
        'a1_depth': (torch.max(gt_depth / (pred_depth + 1e-10), pred_depth / (gt_depth + 1e-10)) < 1.25).float().sum() / (depth_mask.sum() + 1e-10),
    })
    
    return metrics

def visualize_results(sample_idx, batch, pred_disp, pred_depth, output_dir):
    """
    Visualize and save the prediction results
    
    Args:
        sample_idx: Sample index
        batch: Dictionary containing left image, right image, etc.
        pred_disp: Predicted disparity map
        pred_depth: Predicted depth map
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy for visualization
    left_img = batch['left_image'][0].permute(1, 2, 0).cpu().numpy()
    pred_disp_np = pred_disp[0, 0].cpu().numpy()
    pred_depth_np = pred_depth[0].cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot left image
    axes[0].imshow(left_img)
    axes[0].set_title('Left Image')
    axes[0].axis('off')
    
    # Plot predicted disparity
    disp_plot = axes[1].imshow(pred_disp_np, cmap='plasma')
    axes[1].set_title('Predicted Disparity')
    axes[1].axis('off')
    fig.colorbar(disp_plot, ax=axes[1])
    
    # Plot predicted depth (with reasonable depth range)
    depth_plot = axes[2].imshow(np.clip(pred_depth_np, 0, 80), cmap='viridis')
    axes[2].set_title('Predicted Depth')
    axes[2].axis('off')
    fig.colorbar(depth_plot, ax=axes[2])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_{sample_idx:04d}.png'), dpi=200)
    plt.close()
    
    # Also save individual images
    cv2.imwrite(os.path.join(output_dir, f'disp_{sample_idx:04d}.png'), 
                (pred_disp_np / pred_disp_np.max() * 255).astype(np.uint8))
    
    depth_vis = np.clip(pred_depth_np, 0, 80)
    depth_vis = (depth_vis / 80 * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(os.path.join(output_dir, f'depth_{sample_idx:04d}.png'), depth_vis)

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create directory to save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define transformations
    transform = transforms.Compose([
        Resize((args.resize[0], args.resize[1])),
        ToTensor(),
        Normalize()
    ])
    
    # Create dataset and data loader
    dataset = KITTIDataset(root_dir=args.data_dir, split='val', transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load model
    model = StereoTransformerNet().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {args.model_path}")
    
    # Evaluation
    metrics_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            if args.max_samples is not None and i >= args.max_samples:
                break
                
            # Move data to device
            left_img = batch['left_image'].to(device)
            right_img = batch['right_image'].to(device)
            gt_disp = batch['disparity'].to(device) if 'disparity' in batch else None
            focal = batch['focal_length'].to(device)
            baseline = batch['baseline'].to(device)
            
            # Forward pass
            pred_disp = model(left_img, right_img)
            
            # Calculate depth
            pred_depth = torch.zeros_like(pred_disp[:, 0])
            for b in range(pred_disp.size(0)):
                pred_depth[b] = (focal[b] * baseline[b]) / (pred_disp[b, 0] + 1e-10)
                pred_depth[b] = torch.clamp(pred_depth[b], min=0, max=80)
            
            # Visualize results
            visualize_results(i, batch, pred_disp, pred_depth, args.output_dir)
            
            # Calculate metrics if ground truth is available
            if gt_disp is not None:
                batch_metrics = calculate_metrics(pred_disp, gt_disp, focal, baseline)
                metrics_list.append({k: v.item() for k, v in batch_metrics.items()})
    
    # Aggregate and print metrics
    if metrics_list:
        avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
        
        print("\nEvaluation Metrics:")
        print("-" * 50)
        print(f"Disparity MAE: {avg_metrics['mae_disp']:.4f}")
        print(f"Disparity RMSE: {avg_metrics['rmse_disp']:.4f}")
        print(f"Disparity Abs Rel: {avg_metrics['abs_rel_disp']:.4f}")
        print(f"Disparity <1.25: {avg_metrics['a1_disp']:.4f}")
        print("-" * 50)
        print(f"Depth MAE: {avg_metrics['mae_depth']:.4f}")
        print(f"Depth RMSE: {avg_metrics['rmse_depth']:.4f}")
        print(f"Depth Abs Rel: {avg_metrics['abs_rel_depth']:.4f}")
        print(f"Depth <1.25: {avg_metrics['a1_depth']:.4f}")
        
        # Save metrics to file
        with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == '__main__':
    main()