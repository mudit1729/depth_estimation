# Stereo Depth Estimation with Transformers

This module implements a deep learning approach for stereo depth estimation using a transformer-based architecture.

## Architecture

The network takes a pair of stereo images (left and right) and outputs a disparity map, which can be converted to a depth map using the standard stereo formula:

```
depth = (focal_length * baseline) / disparity
```

The architecture consists of:
1. A CNN backbone to extract features from both images
2. Feature concatenation to form a cost volume
3. Transformer encoder layers to fuse spatial context
4. A regression head to predict disparity values

## Usage

### Training

To train the model on the KITTI dataset:

```bash
python train.py --data_dir /path/to/kitti --batch_size 4 --num_epochs 20
```

Important parameters:
- `--data_dir`: Path to the KITTI dataset directory
- `--batch_size`: Batch size for training (default: 4)
- `--num_epochs`: Number of training epochs (default: 20)
- `--lr`: Learning rate (default: 0.0001)
- `--resize`: Image resize dimensions (default: 256 512)
- `--save_dir`: Directory to save checkpoints (default: 'checkpoints')

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --data_dir /path/to/kitti --model_path checkpoints/best_model.pth
```

Important parameters:
- `--data_dir`: Path to the KITTI dataset
- `--model_path`: Path to the trained model checkpoint
- `--output_dir`: Directory to save evaluation results (default: 'results')
- `--max_samples`: Maximum number of samples to evaluate (default: all)

## Requirements

- PyTorch
- torchvision
- NumPy
- matplotlib
- opencv-python
- tqdm