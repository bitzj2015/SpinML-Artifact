# Sample dataset for reproducing privacy and utility results
A sample dataset of Husky images can be downloaded from the link given below:

Sequeira, R., Zhang, J., & Konstantinos, P. (2025). SpinML: Customized Synthetic Data Generation for Private Training of Specialized ML Models [Data set]. SpinML: Customized Synthetic Data Generation for Private Training of Specialized ML Models (SpinML), 25th Privacy Enhancing Technologies Symposium (PETS 2025). Zenodo. https://doi.org/10.5281/zenodo.14777572

# SpinML Sample Dataset Overview

This dataset is used for experiments in the **SpinML** paper. It consists of images of **Husky dogs**, classified based on their activities such as **sitting, sleeping, eating, and playing**.

The dataset includes both **real** and **synthetic** images, allowing for an evaluation of the **privacy-utility trade-off**, as discussed in the paper.

---

## Dataset Structure

The dataset is provided as a compressed **.tar.gz** file. Once extracted, the files should locate in:  
`./SpinML-Artifact/data/husky/`

This directory contains two main subfolders:

### 1. Real Dataset (`real/`)
This folder consists of **actual images** of Huskies and their segmented components. It contains:

- **`raw/`** → Images of Huskies labeled based on their activities.
- **`raw_split/`** → Segmented images with three subfolders:
  - `foreground/` → Extracted Husky images (only the dog).
  - `background/` → The original image with the Husky removed.
  - `mask/` → Binary masks indicating the Husky’s position.

### 2. Synthetic Dataset (`synthetic/`)
This folder contains **AI-generated images** using a **diffusion model**, categorized based on **privacy leakage levels**. The following subfolders represent different levels of privacy exposure:

- **`L0_L0/`** → Foreground and background generated **without any guiding images or features** (No privacy leakage).
- **`L1_L0/`** → Foreground generated using **image features (e.g., Canny edge detection)**, while background remains at **L0** (Partial privacy leakage).
- **`L2_L0/`** → Foreground generated using **the real image as a guideline**, while background remains at **L0** (Full privacy leakage).

Each of these folders contains:
  - `foreground/` → AI-generated Husky images.
  - `background/` → AI-generated backgrounds.
  - `mask/` → Masks for the Husky images.

---

## Privacy Implications
The synthetic dataset allows for studying **privacy-preserving image generation**.
- **L0**: No privacy leakage (no reference to original images).
- **L1**: Partial privacy leakage (uses image features like edges).
- **L2**: Full privacy leakage (directly guided by real images).

This sample dataset is helpful for analyzing the trade-off between **data privacy and model utility** in **SpinML** experiments.
