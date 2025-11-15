# Carotid-Ultrasound-Cascaded-MultiTask-Model

This repository contains the source code for the paper: **A Cascaded Multi-Task Model for Carotid Ultrasound: Cross-Task Attention and Conditional Gating for Plaque Detection, Detail Characterization, and Vulnerability Assessment**.

The model is an end-to-end deep learning framework for analyzing carotid ultrasound images. It performs plaque detection, characterizes five key details (morphology, surface, echo, calcification, stenosis), and assesses plaque vulnerability. Key innovations include Cross-Task Attention Network (CTAN) for multi-task interaction and conditional gating to mimic clinical reasoning.

The code is implemented in PyTorch and supports training, inference, and evaluation on B-mode ultrasound images. It uses YOLOv8 for ROI extraction (in a separate script) and ConvNeXt-base as the backbone.

**Note**: Due to patient privacy and ethical constraints, the full dataset is not included. 

## Features

* End-to-end cascaded multi-task learning.
* CTAN for cross-task feature interaction.
* Conditional gating for vulnerability prediction.
* Focal Loss handling for class imbalance.
* Evaluation metrics: AUC, AP, confusion matrices, ROC/PR curves.
* Visualizations: Grad-CAM, t-SNE, Sankey diagrams (via custom plot scripts).
* ONNX export support for deployment.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/xuqi403/Carotid-Ultrasound-Cascaded-MultiTask-Model.git
   cd Carotid-Ultrasound-Cascaded-MultiTask-Model
   ```

2. Create a virtual environment (Python 3.8+ recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

   * For GPU support, ensure PyTorch is installed with CUDA (e.g., adjust via `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).
   * Optional: If using ONNX export, install `onnx` and `onnx-simplifier`.

## Quick Start

### Data Preparation

* Prepare your dataset in Excel format (e.g., `data.xlsx`) with sheets for train/val/test.
* Columns: `ImagePath` (full path to images), `plaque` (binary), `form`, `surface`, `echo`, `calcification`, `stenosis` (multi-class), `vulnerability` (binary).
* Images: B-mode ultrasound in long-axis view (RGB or convertible).

### Training

Run the training script:

```
python src/train.py --excel_file path/to/data.xlsx --sheet_name Train --batch_size 64 --epochs 50 --output_dir results/
```

* Key arguments:

  * `--excel_file`: Path to Excel with image paths and labels.
  * `--sheet_name`: Sheet for training data (e.g., "Train").
  * `--batch_size`: Default 64.
  * `--epochs`: Default 50 (with warmup for first 3 epochs).
  * `--output_dir`: Where to save models, logs, and plots.
* The script computes dynamic alpha/gamma for Focal Loss based on class distribution.
* Logs: TensorBoard compatible; metrics saved as CSV.

### Inference

```
python infer.py --model_path checkpoints/best_model.pth --image_path path/to/test_image.jpg --output_dir results/infer/
```

* Outputs: Predictions for plaque, details, vulnerability; optional Grad-CAM heatmaps.

### YOLO ROI Extraction

```
python yolo/train.py --data_path path/to/images --epochs 500 --batch_size 8
```

* Pre-trains YOLOv8 for carotid ROI.
* Use in pipeline: Crop ROI before feeding to main model.

* Computes AUC, AP, F1, confusion matrices.
* Generates plots: ROC/PR curves, confusion matrices (via `plot.py`).

## Code Structure

```
├── README.md               
├── LICENSE                 # MIT License
├── requirements.txt        # Dependencies
├── src/                    # Main source code for the cascaded multi-task model
│   ├── model.py            # Model architecture
│   ├── train.py            # Training script with optimizer, scheduler, and multi-task loss handling
│   ├── infer.py            # Inference script for predictions and vulnerability assessment
│   ├── focalloss.py        # Focal Loss implementation
│   ├── loss.py             # Custom loss functions 
│   ├── data.py             # Data loading and dataset classes 
│   ├── utils.py            # Utilities: Seed setting, helper functions, and miscellaneous tools
│   └── plot.py             # Plotting functions for confusion matrices, ROC/PR curves, and visualizations
├── yolo/                   # YOLO-related code for ROI extraction
│   ├── predict.py          # Prediction script for YOLO ROI detection
│   ├── train.py            # Training script for YOLO model
│   └── yolo.py             # Main YOLO utilities and entry points
│   ├── nets/               # YOLO network components
│   │   ├── __init__.py     # Package initializer
│   │   ├── backbone.py     # Backbone network for YOLO
│   │   ├── yolo_training.py# YOLO training modules and heads
│   │   └── yolo.py         # Core YOLO model definition
│   └── utils/              # YOLO utility functions
│       ├── __init__.py     # Package initializer
│       ├── callbacks.py    # Callbacks for training
│       ├── dataloader.py   # Data loaders for YOLO datasets
│       ├── utils_bbox.py   # Bounding box utilities
│       ├── utils_fig.py    # Figure and visualization helpers
│       ├── utils_map.py    # mAP calculation and evaluation
│       └── utils.py        # General utilities 
```

## Model Details

* **Backbone**: ConvNeXt-base (pre-trained on ImageNet).
* **CTAN**: Double-layer cross-task attention for 5 detail tasks.
* **Conditional Gating**: SE-gate + detail probabilities for vulnerability.
* **Loss**: Focal Loss (γ=3.0) for plaque/details; BCE for vulnerability.
* **Optimizer**: AdamW with grouped LR (backbone: 1e-4, heads: 1e-3).
* **Scheduler**: StepLR (decay 0.1 every 5 epochs).
* **Ablations**: See paper for backbone comparisons (ResNet, DenseNet, etc.).

For reproducibility, set seed=42. Training on 2x NVIDIA RTX 4090 takes ~hours (adjust batch_size for memory).

## Limitations

* Dataset not included (privacy reasons).
* Tested on specific hardware; adjust for your setup.
* No real-time integration; extend for clinical use.

## Citation

If you use this code, please cite the paper:

```
@article{carotid_multitask_2025,
  title={A Cascaded Multi-Task Model for Carotid Ultrasound: Cross-Task Attention and Conditional Gating for Plaque Detection, Detail Characterization, and Vulnerability Assessment},
  author={},
  journal={},
  year={2025},
  url={}
}
```

## Contributing

Contributions are welcome! Please open an issue or pull request for bugs, features, or improvements. For major changes, discuss first.

Contact: xuqi403@gmail.com
