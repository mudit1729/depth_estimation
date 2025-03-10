import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Import local modules
from models import StereoTransformerNet
from dataset import KITTIDataset, ToTensor, Normalize, Resize
from loss import DisparityLoss, SmoothL1DisparityLoss

# Add numpy import for dataset
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train a stereo depth estimation model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to KITTI dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--resize', type=int, nargs=2, default=[256, 512], help='Resize input images (height, width)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for tensorboard logs')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=1, help='Checkpoint saving interval in epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of training data to use for validation')
    
    return parser.parse_args()

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, log_interval, writer=None):
    model.train()
    running_loss = 0.0
    samples_with_gt = 0
    total_samples = 0
    epoch_loss = 0.0
    step_count = 0
    
    with tqdm(train_loader, desc=f'Epoch {epoch+1}') as pbar:
        for i, batch in enumerate(pbar):
            # Move data to device
            left_img = batch['left_image'].to(device)
            right_img = batch['right_image'].to(device)
            
            # Debug: Print batch info for first batch
            if i == 0 and epoch == 0:
                print(f"Batch keys: {batch.keys()}")
                
            # Check if ground truth disparity is available
            if 'disparity' in batch:
                gt_disp = batch['disparity'].to(device)
                samples_with_gt += batch['left_image'].size(0)
                
                # Debug first batch
                if i == 0 and epoch == 0:
                    print(f"GT disparity shape: {gt_disp.shape}")
                    print(f"GT disparity min: {gt_disp.min().item()}, max: {gt_disp.max().item()}")
            else:
                if i == 0 and epoch == 0:
                    print("WARNING: No disparity ground truth in batch!")
                gt_disp = None
            
            total_samples += batch['left_image'].size(0)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_disp = model(left_img, right_img)
            
            # Debug first prediction
            if i == 0 and epoch == 0:
                print(f"Prediction shape: {pred_disp.shape}")
                print(f"Prediction min: {pred_disp.min().item()}, max: {pred_disp.max().item()}")
            
            # Skip if no ground truth
            if gt_disp is None:
                if i == 0:
                    print("Skipping batch with no ground truth")
                continue
            
            # Compute loss
            loss = criterion(pred_disp, gt_disp, left_img)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            step_count += 1
            
            # Log to TensorBoard
            global_step = epoch * len(train_loader) + i
            if writer is not None:
                writer.add_scalar('train/batch_loss', loss.item(), global_step)
                
                # Log images periodically (first batch of epoch)
                if i == 0:
                    # Log input images
                    writer.add_images('train/left_image', left_img[:4], global_step)
                    writer.add_images('train/right_image', right_img[:4], global_step)
                    
                    # Log ground truth and prediction
                    writer.add_images('train/gt_disparity', 
                                     gt_disp[:4] / gt_disp.max(), 
                                     global_step)
                    writer.add_images('train/pred_disparity', 
                                     pred_disp[:4] / pred_disp.max(), 
                                     global_step)
            
            # Update progress bar
            if (i + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                pbar.set_postfix(loss=avg_loss)
                running_loss = 0.0
    
    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / step_count if step_count > 0 else float('inf')
    
    # Log epoch stats
    if writer is not None:
        writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
        writer.add_scalar('train/samples_with_gt', samples_with_gt, epoch)
    
    # Print stats after epoch
    if epoch == 0:
        print(f"Epoch stats: {samples_with_gt}/{total_samples} samples had ground truth disparity")
        
    return avg_epoch_loss
    
def validate(model, val_loader, criterion, device, epoch=0, writer=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    # Metrics for disparity error
    abs_rel_error = 0.0
    rmse_error = 0.0
    d1_outlier_sum = 0.0  # D1 metric - percentage of disparity outliers
    d2_outlier_sum = 0.0  # D2 metric - more strict outlier threshold
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc='Validating')):
            # Move data to device
            left_img = batch['left_image'].to(device)
            right_img = batch['right_image'].to(device)
            gt_disp = batch['disparity'].to(device) if 'disparity' in batch else None
            
            # Skip samples without ground truth
            if gt_disp is None:
                print("Warning: No ground truth disparity found for validation. Check dataset setup.")
                continue
            
            # Forward pass
            pred_disp = model(left_img, right_img)
            
            # Compute loss
            loss = criterion(pred_disp, gt_disp, left_img)
            
            # Compute metrics for disparity
            valid_mask = (gt_disp > 0).float()
            abs_diff = torch.abs(pred_disp - gt_disp) * valid_mask
            
            # Absolute relative error
            abs_rel = (abs_diff / (gt_disp + 1e-10) * valid_mask).sum() / (valid_mask.sum() + 1e-10)
            
            # RMSE
            rmse = torch.sqrt((abs_diff ** 2).sum() / (valid_mask.sum() + 1e-10))
            
            # D1 & D2 metrics - percentage of stereo disparity outliers
            # An outlier is defined as a pixel where the disparity error is:
            # - greater than 3 pixels, OR
            # - greater than 5% of the ground truth value
            
            # D1 - percentage of disparity outliers in first frame
            d1_outlier_condition = ((abs_diff > 3.0) | (abs_diff > 0.05 * torch.abs(gt_disp))) & (valid_mask > 0)
            d1_outlier_percentage = (d1_outlier_condition.float().sum() / (valid_mask.sum() + 1e-10)) * 100.0
            
            # D2 would be for second frame disparity, but we're focused on the first frame here
            # For completeness, we'll calculate another variant:
            # More strict: outliers with error > 2 pixels OR > 3% of ground truth
            d2_outlier_condition = ((abs_diff > 2.0) | (abs_diff > 0.03 * torch.abs(gt_disp))) & (valid_mask > 0)
            d2_outlier_percentage = (d2_outlier_condition.float().sum() / (valid_mask.sum() + 1e-10)) * 100.0
            
            # Update statistics
            batch_size = left_img.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            abs_rel_error += abs_rel.item() * batch_size
            rmse_error += rmse.item() * batch_size
            d1_outlier_sum += d1_outlier_percentage.item() * batch_size
            d2_outlier_sum += d2_outlier_percentage.item() * batch_size
            
            # Log first batch to TensorBoard
            if i == 0 and writer is not None:
                # Log images
                writer.add_images('val/left_image', left_img[:4], epoch)
                writer.add_images('val/right_image', right_img[:4], epoch)
                
                # Log ground truth and prediction
                writer.add_images('val/gt_disparity', 
                                 gt_disp[:4] / (gt_disp[:4].max() + 1e-5), 
                                 epoch)
                writer.add_images('val/pred_disparity', 
                                 pred_disp[:4] / (pred_disp[:4].max() + 1e-5), 
                                 epoch)
                
                # Log error maps
                error_map = torch.abs(pred_disp - gt_disp) * valid_mask
                error_map = error_map / (error_map.max() + 1e-5)
                writer.add_images('val/error_map', error_map[:4], epoch)
    
    # Compute average loss and metrics
    if total_samples == 0:
        print("ERROR: No validation samples with ground truth disparity were found!")
        print("Using a default loss value. Please check your dataset configuration.")
        avg_loss = 1000.0     # A high default value rather than infinity
        avg_abs_rel = 1.0     # Default high value
        avg_rmse = 100.0      # Default high value
        avg_d1_outlier = 100.0  # Default: 100% outliers
        avg_d2_outlier = 100.0  # Default: 100% outliers
    else:
        avg_loss = total_loss / total_samples
        avg_abs_rel = abs_rel_error / total_samples
        avg_rmse = rmse_error / total_samples
        avg_d1_outlier = d1_outlier_sum / total_samples
        avg_d2_outlier = d2_outlier_sum / total_samples
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/abs_rel_error', avg_abs_rel, epoch)
        writer.add_scalar('val/rmse', avg_rmse, epoch)
        writer.add_scalar('val/d1_outlier_percentage', avg_d1_outlier, epoch)
        writer.add_scalar('val/d2_outlier_percentage', avg_d2_outlier, epoch)
        
    # Return average values
    metrics = {
        'loss': avg_loss,
        'abs_rel': avg_abs_rel,
        'rmse': avg_rmse,
        'd1_outlier': avg_d1_outlier,
        'd2_outlier': avg_d2_outlier
    }
    
    return metrics

def save_checkpoint(model, optimizer, epoch, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

def visualize_predictions(model, val_loader, device, output_dir, num_samples=5):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
                
            # Move data to device
            left_img = batch['left_image'].to(device)
            right_img = batch['right_image'].to(device)
            gt_disp = batch['disparity'].to(device) if 'disparity' in batch else None
            
            # Forward pass
            pred_disp = model(left_img, right_img)
            
            # Convert to depth using the focal length and baseline
            focal = batch['focal_length'].to(device)
            baseline = batch['baseline'].to(device)
            
            # For visualization, we need to handle NaN and Inf values
            pred_disp_vis = torch.clamp(pred_disp[0, 0], min=0.1)  # Avoid division by zero
            pred_depth = (focal[0] * baseline[0]) / pred_disp_vis
            
            # Clamp depth values to a reasonable range for visualization
            pred_depth = torch.clamp(pred_depth, min=0, max=80)
            
            # Convert tensors to numpy for visualization
            left_img_np = batch['left_image'][0].permute(1, 2, 0).cpu().numpy()
            pred_disp_np = pred_disp[0, 0].cpu().numpy()
            pred_depth_np = pred_depth.cpu().numpy()
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot left image
            axes[0].imshow(left_img_np)
            axes[0].set_title('Left Image')
            axes[0].axis('off')
            
            # Plot predicted disparity
            disp_plot = axes[1].imshow(pred_disp_np, cmap='plasma')
            axes[1].set_title('Predicted Disparity')
            axes[1].axis('off')
            fig.colorbar(disp_plot, ax=axes[1])
            
            # Plot predicted depth
            depth_plot = axes[2].imshow(pred_depth_np, cmap='viridis')
            axes[2].set_title('Predicted Depth')
            axes[2].axis('off')
            fig.colorbar(depth_plot, ax=axes[2])
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i}.png'))
            plt.close()

def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create TensorBoard log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, f'stereo_depth_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    print(f"View logs with: tensorboard --logdir={args.log_dir}")
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Define transformations
    train_transform = transforms.Compose([
        Resize((args.resize[0], args.resize[1])),
        ToTensor(),
        Normalize()
    ])
    
    val_transform = transforms.Compose([
        Resize((args.resize[0], args.resize[1])),
        ToTensor(),
        Normalize()
    ])
    
    # Create datasets and data loaders
    train_dataset = KITTIDataset(
        root_dir=args.data_dir, 
        split='train', 
        transform=train_transform, 
        val_ratio=args.val_ratio
    )
    
    val_dataset = KITTIDataset(
        root_dir=args.data_dir, 
        split='val', 
        transform=val_transform,
        val_ratio=args.val_ratio
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Create model
    model = StereoTransformerNet().to(device)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_trainable_params} trainable parameters")
    
    # Log model architecture to TensorBoard
    writer.add_text('model/architecture', str(model), 0)
    writer.add_text('model/parameters', f"Trainable parameters: {num_trainable_params}", 0)
    
    # Define loss and optimizer
    criterion = DisparityLoss(weights={'l1': 1.0, 'smooth': 0.1})
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Log hyperparameters
    writer.add_hparams(
        {
            'lr': args.lr,
            'batch_size': args.batch_size,
            'image_height': args.resize[0],
            'image_width': args.resize[1],
            'val_ratio': args.val_ratio,
            'model': 'StereoTransformerNet'
        },
        {}  # Metrics dict (will be filled during training)
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch, args.log_interval, writer
        )
        train_losses.append(train_loss)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer)
        val_loss = val_metrics['loss']
        val_losses.append(val_loss)
        
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Metrics:")
        print(f"  - Abs Rel: {val_metrics['abs_rel']:.4f}")
        print(f"  - RMSE: {val_metrics['rmse']:.4f}")
        print(f"  - D1 (>3px or >5%): {val_metrics['d1_outlier']:.2f}%")
        print(f"  - D2 (>2px or >3%): {val_metrics['d2_outlier']:.2f}%")
        
        # Save checkpoint if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))
    
    # Plot training and validation loss history
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.num_epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, args.num_epochs + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'loss_history.png'))
    
    # Add the plot to TensorBoard
    writer.add_figure('Loss/history', plt.gcf(), global_step=args.num_epochs)
    
    # Generate visualizations
    vis_dir = os.path.join(args.save_dir, f'vis_{timestamp}')
    visualize_predictions(model, val_loader, device, vis_dir)
    print(f"Saved visualizations to {vis_dir}")
    
    # Save final model
    save_checkpoint(model, optimizer, args.num_epochs-1, os.path.join(args.save_dir, 'final_model.pth'))
    
    # Close TensorBoard writer
    writer.close()
    
    print("Training complete!")
    print(f"View TensorBoard logs with: tensorboard --logdir={args.log_dir}")

if __name__ == '__main__':
    main()