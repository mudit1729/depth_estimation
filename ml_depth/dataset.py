import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image

class KITTIDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, val_ratio=0.2, random_seed=42):
        """
        KITTI Stereo Dataset
        
        Args:
            root_dir (string): Directory with the KITTI dataset
            split (string): 'train' or 'val' split
            transform (callable, optional): Optional transform to be applied on samples
            val_ratio (float): Ratio of training data to use for validation (default: 0.2)
            random_seed (int): Random seed for reproducible train/val split
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Get all file paths from training data
        all_left_images = sorted(glob.glob(os.path.join(root_dir, 'training', 'image_2', '*_10.png')))
        all_right_images = sorted(glob.glob(os.path.join(root_dir, 'training', 'image_3', '*_10.png')))
        all_disp_maps = sorted(glob.glob(os.path.join(root_dir, 'training', 'disp_noc_0', '*.png')))
        
        # If no disparity maps are found, try to find them in the expected location
        if not all_disp_maps:
            print(f"Warning: No disparity maps found in {os.path.join(root_dir, 'training', 'disp_noc_0')}.")
            # Try alternative locations
            alternative_paths = [
                os.path.join(root_dir, 'training', 'disp_occ_0'),
                os.path.join(root_dir, 'training', 'disp_noc'),
                os.path.join(root_dir, 'training', 'disp')
            ]
            for path in alternative_paths:
                all_disp_maps = sorted(glob.glob(os.path.join(path, '*.png')))
                if all_disp_maps:
                    print(f"Found disparity maps in {path}")
                    break
            
            # If still no disparity maps, use a dummy approach for testing
            if not all_disp_maps:
                print("Warning: No disparity maps found. Using left images as dummy disparity maps for testing.")
                all_disp_maps = all_left_images.copy()
        
        # Ensure all lists have the same length
        min_len = min(len(all_left_images), len(all_right_images), len(all_disp_maps))
        all_left_images = all_left_images[:min_len]
        all_right_images = all_right_images[:min_len]
        all_disp_maps = all_disp_maps[:min_len]
        
        # Create train/val split
        np.random.seed(random_seed)
        indices = np.arange(len(all_left_images))
        np.random.shuffle(indices)
        
        val_size = int(len(indices) * val_ratio)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        if split == 'train':
            # Use training portion
            self.left_images = [all_left_images[i] for i in train_indices]
            self.right_images = [all_right_images[i] for i in train_indices]
            self.disp_maps = [all_disp_maps[i] for i in train_indices]
        elif split == 'val':
            # Use validation portion
            self.left_images = [all_left_images[i] for i in val_indices]
            self.right_images = [all_right_images[i] for i in val_indices]
            self.disp_maps = [all_disp_maps[i] for i in val_indices]
        elif split == 'test':
            # For actual testing, use the testing set (no ground truth)
            self.left_images = sorted(glob.glob(os.path.join(root_dir, 'testing', 'image_2', '*_10.png')))
            self.right_images = sorted(glob.glob(os.path.join(root_dir, 'testing', 'image_3', '*_10.png')))
            self.disp_maps = None
        
        # Get camera calibration parameters for each scene
        self.focal_lengths = []
        self.baselines = []
        
        for left_img_path in self.left_images:
            # Extract sequence number from filename
            seq_num = os.path.basename(left_img_path).split('_')[0]
            
            # Determine which calibration file to use
            # For both train and val splits, use the training calibration
            # For test split, use the testing calibration
            if split == 'test':
                calib_folder = 'testing'
            else:
                calib_folder = 'training'
                
            calib_file = os.path.join(root_dir, calib_folder, 'calib_cam_to_cam', f'{seq_num}.txt')
            
            # Extract focal length and baseline from calibration file
            focal, baseline = self._read_calibration(calib_file)
            self.focal_lengths.append(focal)
            self.baselines.append(baseline)
    
    def _read_calibration(self, calib_file):
        """Read calibration parameters from KITTI calibration file"""
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        
        # Default values in case we can't find the calibration
        focal = 721.5377  # Default KITTI focal length
        baseline = 0.54   # Default KITTI baseline
        
        for line in lines:
            if line.startswith('P2:'):  # Camera intrinsics for left camera
                P2 = np.array(line.strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
                focal = P2[0, 0]  # Focal length is P2[0,0]
            
            if line.startswith('P3:'):  # Camera intrinsics for right camera
                P3 = np.array(line.strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
                # The baseline is the distance between camera centers, found in the translation part
                baseline = abs(P3[0, 3] / P3[0, 0])
                
        return focal, baseline
    
    def __len__(self):
        return len(self.left_images)
    
    def __getitem__(self, idx):
        # Load images
        left_img = cv2.imread(self.left_images[idx])
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        
        right_img = cv2.imread(self.right_images[idx])
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        
        # Get calibration data
        focal = self.focal_lengths[idx]
        baseline = self.baselines[idx]
        
        # Create sample dict
        sample = {
            'left_image': left_img,
            'right_image': right_img,
            'focal_length': focal,
            'baseline': baseline
        }
        
        # Load ground truth disparity map if available (only for training data)
        if self.split == 'train' and self.disp_maps is not None:
            # KITTI disparity maps are stored as uint16 PNG with a scale factor of 256
            disp_map = cv2.imread(self.disp_maps[idx], cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            sample['disparity'] = disp_map
        
        # Apply transformations
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ToTensor(object):
    """Convert numpy arrays in sample to PyTorch tensors."""
    
    def __call__(self, sample):
        # Convert images from HWC to CHW format
        left_image = sample['left_image'].transpose((2, 0, 1))
        right_image = sample['right_image'].transpose((2, 0, 1))
        
        # Create output sample with tensors
        output = {
            'left_image': torch.from_numpy(left_image).float() / 255.0,  # Normalize to [0, 1]
            'right_image': torch.from_numpy(right_image).float() / 255.0,
            'focal_length': torch.tensor(sample['focal_length'], dtype=torch.float),
            'baseline': torch.tensor(sample['baseline'], dtype=torch.float)
        }
        
        # Add disparity if available
        if 'disparity' in sample:
            disparity = sample['disparity']
            # Add a channel dimension if needed
            if len(disparity.shape) == 2:
                disparity = disparity[None, :, :]  # Add channel dimension
            else:
                disparity = disparity.transpose((2, 0, 1))
            output['disparity'] = torch.from_numpy(disparity).float()
        
        return output


class Normalize(object):
    """Normalize images with mean and standard deviation."""
    
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        # Clone the sample to avoid modifying the original
        output = sample.copy()
        
        # Normalize left and right images
        for img_key in ['left_image', 'right_image']:
            if isinstance(sample[img_key], torch.Tensor):
                mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
                std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
                output[img_key] = (sample[img_key] - mean) / std
            else:
                # For numpy arrays (shouldn't happen after ToTensor)
                mean = np.array(self.mean, dtype=np.float32).reshape(3, 1, 1)
                std = np.array(self.std, dtype=np.float32).reshape(3, 1, 1)
                output[img_key] = (sample[img_key] - mean) / std
        
        return output


class Resize(object):
    """Resize images and disparity maps."""
    
    def __init__(self, size):
        """
        Args:
            size (tuple): Desired output size (height, width)
        """
        self.size = size
    
    def __call__(self, sample):
        h, w = self.size
        
        # Resize left and right images
        left_img_resized = cv2.resize(sample['left_image'], (w, h), interpolation=cv2.INTER_LINEAR)
        right_img_resized = cv2.resize(sample['right_image'], (w, h), interpolation=cv2.INTER_LINEAR)
        
        output = {
            'left_image': left_img_resized,
            'right_image': right_img_resized,
            'focal_length': sample['focal_length'] * (w / sample['left_image'].shape[1]),  # Scale focal length
            'baseline': sample['baseline']
        }
        
        # Resize disparity map if available, scaling the values to match the new size
        if 'disparity' in sample:
            scale_factor = w / sample['left_image'].shape[1]
            disp_resized = cv2.resize(sample['disparity'], (w, h), interpolation=cv2.INTER_LINEAR)
            # Scale the disparity values to match the new image size
            disp_resized = disp_resized * scale_factor
            output['disparity'] = disp_resized
        
        return output