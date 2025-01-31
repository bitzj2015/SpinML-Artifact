# Use synthetic data to train specialized ML models

# Required libraries
To run the scripts, ensure the following libraries are installed:
```
pandas==1.5.2
pillow==10.2.0
torch==2.1.0
torchvision==0.16.0
tqdm==4.65.2
```

# Train a specialized MobileNet
```
python train_mobilenet.py --alpha $alpha --random_seed $seed --testdata $testdata --augdata $augdata
```
Example:
```
./train_mobilenet_example.sh
```

# Dataset Labeling and YOLO Training Pipeline
This folder contains three scripts designed for `YOLO` model training:
1. **Label a synthetic dataset** using the `GroundedSAM` model.
2. **Train and validate YOLO models** on the labeled dataset.
3. **Run a full pipeline** to label datasets and train YOLO models sequentially.

Ensure you have the following packages installed:

- Python 3.10+
- To install the `ultralytics` package from its GitHub repository for the latest updates and details, follow the instructions provided in their [GitHub repository](https://github.com/ultralytics/ultralytics).
  ```bash
  git clone https://github.com/ultralytics/ultralytics.git
  cd ultralytics
  pip install -e .
  ```
- Autodistill

  Autodistill is modular. You'll need to install the autodistill package (which defines the interfaces for the above concepts) along with Base Model and Target Model plugins (which implement specific models).
  
  By packaging these separately as plugins, dependency and licensing incompatibilities are minimized and new models can be implemented and maintained by anyone.
  
  Example:
    ```bash
    pip install autodistill autodistill-grounded-sam autodistill-yolov8
    ```
  
  Please refer to autodistill github page for other ways to install the package.

## Scripts

### 1. `data_utils_yolo.py`
This script labels images in a specified folder using the `GroundedSAM` model.

#### Arguments:
- `--input_folder`: Path to the input folder containing images.
- `--output_folder`: Path to save the labeled dataset.
- `--prompt`: Prompt to find objects (default: `"pill bottle"`).
- `--label`: Label for found objects (default: `"bottle"`).
- `--extension`: Image file extension (default: `.png`).

#### Example:
```bash
python data_utils_yolo.py \
  --input_folder /path/to/images \
  --output_folder /path/to/labeled_data \
  --prompt "pill bottle" \
  --label "bottle" \
  --extension .png
```

### 2. `train_utils_yolo.py`
This script trains and validates a YOLO model on the labeled dataset.

#### Arguments:
- `--out_folder`: Path to the dataset folder.
- `--save_folder`: Path to save trained models and results.
- `--yolo_model`: Path to the YOLO model file (e.g., `yolov8n.pt`).
- `--device`: Device to use for training (default: `"cuda:0"`).
- `--epochs`: Number of training epochs (default: `200`).
- `--seed`: Random seed for reproducibility (default: random).

#### Example:
```bash
python train_utils_yolo.py \
  --out_folder /path/to/labeled_data \
  --save_folder /path/to/models \
  --yolo_model yolov8n.pt \
  --device cuda:0 \
  --epochs 200
```

### 3. `train_yolo.py`
This script integrates the labeling and YOLO training processes into a unified pipeline.

#### Arguments:
- `--input_folder`: Path to the input image folder.
- `--output_folder`: Path to save the labeled dataset.
- `--prompt`: Prompt to find objects (default: `"pill bottle"`).
- `--label`: Label for found objects (default: `"bottle"`).
- `--extension`: Image file extension (default: `.png`).
- `--save_folder`: Path to save trained models and results.
- `--yolo_model`: Path to the YOLO model file (e.g., `yolov8n.pt`).
- `--device`: Device to use for training (default: `"cuda:0"`).
- `--epochs`: Number of training epochs (default: `200`).

#### Example:
```bash
python train_yolo.py \
  --input_folder /path/to/images \
  --output_folder /path/to/labeled_data \
  --prompt "pill bottle" \
  --label "bottle" \
  --extension .png \
  --save_folder /path/to/models \
  --yolo_model yolov8n.pt \
  --device cuda:0 \
  --epochs 200
```

## Notes
- Adjust paths to match your directory structure.
- Ensure the GPU is available and specified correctly using the `--device` argument.

## Author
This pipeline was developed to streamline dataset labeling and YOLO model training workflows.

