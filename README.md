# SpinML-Artifact

Paper title: **SpinML: Customized Synthetic Data Generation for Private Training of Specialized ML Models**

## Description
This artifact contains the scripts for running experiments described in this paper.

### Hardware Requirements
We conducted experiments on a server equipped with:

CPU: `AMD Threadripper 3970X`
GPU: Two `NVIDIA RTX A6000` (each with 48GB memory)
Memory: 128GB RAM
While these are not strict requirements, we recommend using a server with at least 16GB of GPU memory to successfully reproduce the experiments.

### Software Requirements
The experiments were performed on a server running `Ubuntu 20.04.1 LTS`. Hence, Linux environment is recommended for this project. This is not tested on another operating systems. 

1. Install Python version 3.10. It is recommended to create and install packages in a virtual environment (e.g: conda). 

  ```
  conda create -n spinml python=3.10 -y
  conda activate spinml
  pip install --upgrade pip 
  ```

2. Install the following Python libraries which are required to run scripts under `src/privacy/` folder:

  ```
  numpy==1.24.4
  pillow==10.2.0
  torch==2.1.0
  torchvision==0.16.0
  tqdm==4.65.2
  transformers==4.46.3
  controlnet-aux==0.0.7
  ```

3. Install the following additional Python libraries which are required to run various scripts under `src/utility/` folder:

  ```
  pandas==1.5.2
  pillow==10.2.0
  torch==2.1.0
  torchvision==0.16.0
  tqdm==4.65.2
  autodistill==0.1.29
  autodistill-grounded-sam==0.1.2
  autodistill-yolov8==0.1.4
  roboflow==1.1.50
  scikit-learn==1.6.1
  ```
  
  Note: Autodistill is modular. You'll need to install the autodistill package along with Base Model and Target Model plugins (which implement specific models). Please refer to autodistill github page for other ways to install the package and its required packages.

4. Install the additional `ultralytics` package for the YOLO Training Pipeline under `src/utility/` folder:

  ```
  git clone https://github.com/ultralytics/ultralytics.git
  cd ultralytics
  pip install -e .
  ```

  Note: install the `ultralytics` package from its GitHub repository for the latest updates and details, follow the instructions provided in their [GitHub repository](https://github.com/ultralytics/ultralytics).

### Purpose of Each Directory
1. The folder `src/privacy` contains scripts for running both MI and SIM privacy measurement described in the paper. Please see `src/privacy/README.md` for details.

2. The folder `src/utility` contains scripts for training both MobileNet and YOLO models described in the paper. Please see `src/utility/README.md` for details.

3. The folder `src/diffusion_model` contains instructions on how to use diffusion models for synthetic data generation. Please see `src/diffusion_model/README.md` for details.

4. The folder `data` contains a sample husky dataset for reproducing privacy and utility results. Please see `data/README.md` for details.