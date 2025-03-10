import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image

class KITTIDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        KITTI Stereo Dataset
        
        Args:
            root_dir (string): Directory with the KITTI dataset
            split (string): 'train' or 'val' split
            transform (callable, optional): Optional transform to be applied on samples
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Get file paths for images and disparity maps
        if split == 'train':
            # Using training data from KITTI
            self.left_images = sorted(glob.glob(os.path.join(root_dir, 'training', 'image_2', '*_10.png')))
            self.right_images = sorted(glob.glob(os.path.join(root_dir, 'training', 'image_3', '*_10.png')))
            self.disp_maps = sorted(glob.glob(os.path.join(root_dir, 'training', 'disp_noc_0', '*.png')))
        else:
            # Using testing data for validation (note: ground truth is not available for the actual test set)
            self.left_images = sorted(glob.glob(os.path.join(root_dir, 'testing', 'image_2', '*_10.png')))
            self.right_images = sorted(glob.glob(os.path.join(root_dir, 'testing', 'image_3', '*_10.png')))
            self.disp_maps = None
        
        # Get camera calibration parameters for each scene
        self.focal_lengths = []
        self.baselines = []
        
        for left_img_path in self.left_images:
            # Extract sequence number from filename
            seq_num = os.path.basename(left_img_path).split('_')[0]
            
            # Read calibration file
            calib_file = os.path.join(root_dir, 'training' if split == 'train' else 'testing', 
                                     'calib_cam_to_cam', f'{seq_num}.txt')
            
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