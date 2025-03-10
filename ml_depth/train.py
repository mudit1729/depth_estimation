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

# Import local modules
from models import StereoTransformerNet
from dataset import KITTIDataset, ToTensor, Normalize, Resize
from loss import DisparityLoss, SmoothL1DisparityLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Train a stereo depth estimation model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to KITTI dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--resize', type=int, nargs=2, default=[256, 512], help='Resize input images (height, width)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=1, help='Checkpoint saving interval in epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    return parser.parse_args()

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, log_interval):
    model.train()
    running_loss = 0.0
    
    with tqdm(train_loader, desc=f'Epoch {epoch+1}') as pbar:
        for i, batch in enumerate(pbar):
            # Move data to device
            left_img = batch['left_image'].to(device)
            right_img = batch['right_image'].to(device)
            gt_disp = batch['disparity'].to(device) if 'disparity' in batch else None
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_disp = model(left_img, right_img)
            
            # Compute loss
            loss = criterion(pred_disp, gt_disp, left_img)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            
            # Update progress bar
            if (i + 1) % log_interval == 0:
                pbar.set_postfix(loss=running_loss / log_interval)
                running_loss = 0.0
    
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            # Move data to device
            left_img = batch['left_image'].to(device)
            right_img = batch['right_image'].to(device)
            gt_disp = batch['disparity'].to(device) if 'disparity' in batch else None
            
            # Skip samples without ground truth
            if gt_disp is None:
                continue
            
            # Forward pass
            pred_disp = model(left_img, right_img)
            
            # Compute loss
            loss = criterion(pred_disp, gt_disp, left_img)
            
            # Update statistics
            batch_size = left_img.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    # Compute average loss
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    return avg_loss

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
    train_dataset = KITTIDataset(root_dir=args.data_dir, split='train', transform=train_transform)
    val_dataset = KITTIDataset(root_dir=args.data_dir, split='val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Create model
    model = StereoTransformerNet().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Define loss and optimizer
    criterion = DisparityLoss(weights={'l1': 1.0, 'smooth': 0.1})
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train for one epoch
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.log_interval)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))
    
    # Generate visualizations
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    vis_dir = os.path.join(args.save_dir, f'vis_{timestamp}')
    visualize_predictions(model, val_loader, device, vis_dir)
    print(f"Saved visualizations to {vis_dir}")
    
    # Save final model
    save_checkpoint(model, optimizer, args.num_epochs-1, os.path.join(args.save_dir, 'final_model.pth'))
    print("Training complete!")

if __name__ == '__main__':
    main()