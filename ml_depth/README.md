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
- `--log_dir`: Directory for TensorBoard logs (default: 'runs')
- `--val_ratio`: Ratio of training data to use for validation (default: 0.2)

### TensorBoard Monitoring

Training progress can be monitored in real-time using TensorBoard:

```bash
tensorboard --logdir=runs
```

The TensorBoard interface provides:
- Training and validation loss curves
- Disparity metrics (RMSE, Absolute Relative Error, D1, D2)
- Input image visualization
- Ground truth and predicted disparity maps
- Error maps showing prediction accuracy
- Model architecture and hyperparameter information

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

## Evaluation Metrics

The model is evaluated using several disparity and depth metrics:

1. **Disparity Metrics**:
   - **RMSE**: Root Mean Square Error of disparity prediction
   - **MAE**: Mean Absolute Error of disparity prediction
   - **Abs Rel**: Absolute Relative Error
   - **D1**: Percentage of outlier pixels where error > 3px OR > 5% of ground truth (standard KITTI metric)
   - **D2**: Percentage of outlier pixels where error > 2px OR > 3% of ground truth (stricter metric)
   - **δ < 1.25**: Percentage of pixels with ratio of predicted/GT or GT/predicted < 1.25

2. **Depth Metrics**:
   - **RMSE**: Root Mean Square Error of depth prediction
   - **MAE**: Mean Absolute Error of depth prediction
   - **Abs Rel**: Absolute Relative Error
   - **δ < 1.25**: Percentage of pixels with ratio of predicted/GT or GT/predicted < 1.25

All metrics are computed only on valid pixels (those with ground truth disparity available).

## Dataset Handling

The training script automatically splits the KITTI training data into train and validation sets based on the specified `val_ratio`. This ensures that both training and validation data have ground truth disparity maps available.

## Requirements

- PyTorch
- torchvision
- transformers (Hugging Face)
- peft (Parameter-Efficient Fine-Tuning)
- TensorBoard
- NumPy
- matplotlib
- opencv-python
- tqdm

All dependencies can be installed using:
```bash
pip install -r requirements.txt
```