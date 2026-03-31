# RetinexRWKV

Retinexformer with RWKV TimeMix for Low-Light Image Enhancement

## Overview

RetinexRWKV is a deep learning-based approach for low-light image enhancement that combines the Retinex theory with RWKV (Receptance Weighted Key Value) attention mechanisms. This project implements an enhanced version of RetinexFormer, replacing traditional attention with RWKV-7 for improved efficiency and performance.

## Architecture

The model consists of:
- **Illumination Estimator**: Estimates illumination features and maps from input images
- **IGAB (Illumination-Guided Attention Block)**: Uses RWKV-7 attention for feature enhancement guided by illumination information
- **Denoiser**: U-Net style encoder-decoder structure with IGAB blocks at multiple levels
- **Multi-stage Design**: Supports cascaded stages for progressive enhancement

### Key Features
- RWKV-7 attention mechanism for efficient sequence modeling
- Illumination-guided feature enhancement
- Multi-scale processing with skip connections
- Support for large image inference via tiling

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)
- Dependencies:
  - numpy
  - Pillow
  - scikit-image
  - einops
  - tqdm
  - thop
  - rwkvfla

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd RetinexRWKV

# Install dependencies
pip install torch torchvision numpy Pillow scikit-image einops tqdm thop
pip install rwkvfla  # RWKV attention implementation
```

## Project Structure

```
RetinexRWKV/
├── models/
│   ├── archs/
│   │   ├── RetinexFormer_arch.py    # Main model architecture with RWKV-7
│   │   ├── rwkv_7.py                # RWKV-7 implementation
│   │   ├── arch_util.py             # Architecture utilities
│   │   └── layers.py                # Custom layers
│   ├── losses/                      # Loss functions
│   ├── lr_scheduler.py              # Learning rate schedulers
│   └── base_model.py
├── datasets_process/
│   └── prefetch_dataloader.py       # Data loading utilities
├── scoring_program_3/
│   ├── evaluation.py                # Evaluation metrics
│   └── myssim.py                    # SSIM implementation
├── train.py                         # Training script
├── test.py                          # Inference script
├── clean.py                         # Data cleaning utilities
├── config.json                      # Configuration file
└── README.md                        # This file
```

## Usage

### Configuration

Edit `config.json` to set your training parameters:

```json
{
  "model": {
    "name": "RetinexFormer",
    "params": {
      "in_channels": 3,
      "out_channels": 3,
      "n_feat": 16,
      "stage": 1,
      "num_blocks": [1, 2, 2]
    }
  },
  "train": {
    "epochs": 500,
    "batch_size": 32,
    "lr": 0.0005,
    "dataset": {
      "train_dir": "datasets/Train_Low_Light_2025/",
      "val_dir": "datasets/Val_New/"
    },
    "patch_size": 256,
    "num_workers": 4,
    "val_epoch": 50
  }
}
```

### Training

Prepare your dataset in the following structure:
```
datasets/
├── Train_Low_Light_2025/
│   ├── Input/       # Low-light images
│   └── GT/          # Ground-truth normal-light images
└── Val_New/
    ├── Input/
    └── GT/
```

Run training:
```bash
python train.py
```

Training features:
- Progress bars for training and validation
- Automatic best model saving based on SSIM
- Cosine annealing learning rate scheduler with restarts
- PSNR and SSIM validation metrics

### Inference

For single image or batch inference:

```bash
python test.py
```

Modify the paths in `test.py`:
```python
input_dir = "path/to/input/images"
output_dir = "path/to/output/images"
model_path = "./best_model.pth"
```

Features:
- Automatic handling of large images (>2000x3000) via tiling
- Batch processing support
- GPU acceleration

### Model Checkpoints

Pre-trained models are available:
- `best_model_nfeat_12.pth`: Model with 12 feature channels
- `best_model_nfeat_16.pth`: Model with 16 feature channels

Load pre-trained weights by setting in `config.json`:
```json
"pretrained_weights": "best_model_nfeat_16.pth"
```

## Model Specifications

| Model | Feature Channels | Stages | Params (M) | FLOPs (G) |
|-------|-----------------|--------|------------|-----------|
| Small | 12 | 1 | ~0.5 | ~15 |
| Base | 16 | 1 | ~0.8 | ~25 |

*Note: Actual values depend on num_blocks configuration*

## Performance Metrics

The model is evaluated using:
- **PSNR** (Peak Signal-to-Noise Ratio): Measures reconstruction quality
- **SSIM** (Structural Similarity Index): Measures perceptual similarity

Validation runs every `val_epoch` epochs during training.

## Dataset Format

- Input images: PNG format, low-light conditions
- Ground truth: PNG format, normal-light conditions
- Recommended patch size: 256x256 for training
- Images should be paired (same filename in Input and GT folders)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{retinexformer2023,
  title={Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement},
  author={Cai, Yuanhao et al.},
  journal={ICCV},
  year={2023}
}

@article{rwkv2023,
  title={RWKV: Reinventing RNNs for the Transformer Era},
  author={Peng, Bo et al.},
  journal={EMNLP},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Original RetinexFormer implementation
- RWKV attention mechanism
- Contributors and users of this project

## Contact

For issues and questions, please open an issue on the GitHub repository.