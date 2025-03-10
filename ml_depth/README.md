# DINOv2-LoRA Stereo Depth Estimation

This module implements a deep learning approach for stereo depth estimation using DINOv2 vision transformers with LoRA adaptation.

## Architecture

The network takes a pair of stereo images (left and right) and outputs a disparity map, which can be converted to a depth map using the standard stereo formula:

```
depth = (focal_length * baseline) / disparity
```

The architecture consists of:

1. **Dual DINOv2-small Backbone**: Two separate vision transformers process the left and right stereo images independently, extracting rich visual features
2. **LoRA Adapters**: Low-rank adaptation applied to both DINOv2 models, enabling efficient fine-tuning with minimal parameter overhead
3. **Cross-Attention Fusion**: Specialized attention mechanism that correlates features between left and right views to establish stereo correspondence
4. **Self-Attention Refinement**: Two self-attention layers that refine the spatial features with global context
5. **Disparity Head**: Final regression pathway to predict disparity values at each pixel location

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

## Benefits of the Architecture

1. **Leveraging Foundation Models**: Uses the powerful DINOv2 vision transformer that has been pretrained on diverse image datasets, providing robust visual representations
2. **Parameter Efficiency**: LoRA adapters reduce the number of trainable parameters by modifying only a small subset of the network
3. **Stereo Correspondence**: Cross-attention effectively learns to match features between the two views
4. **Global Context**: Transformer architecture captures long-range dependencies that are crucial for consistent depth estimation

## Requirements

- PyTorch
- torchvision
- transformers (Hugging Face)
- peft (Parameter-Efficient Fine-Tuning)
- NumPy
- matplotlib
- opencv-python
- tqdm