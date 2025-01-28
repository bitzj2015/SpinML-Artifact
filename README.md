# SpinML-Artifact

Paper title: **SpinML: Customized Synthetic Data Generation for Private Training of Specialized ML Models**

Artifacts HotCRP Id: **#18** (not your paper Id, but the artifacts id)

Requested Badge: **Functional**

## Description
This artifact contains the scripts for running experiments described in this paper.

### Security/Privacy Issues and Ethical Concerns (All badges)
Our artifact does not hold any risk to the security or privacy of the reviewer's machine. Also, our artifact does not contain malware samples, or something similar, to be analyzed. In addition, there are not any ethical concerns regarding our artifacts here.

## Basic Requirements (Only for Functional and Reproduced badges)
The artifact requires the following minimal hardware and software configurations.

### Hardware Requirements
We conducted experiments on a server equipped with:

CPU: `AMD Threadripper 3970X`
GPU: Two `NVIDIA RTX A6000` (each with 48GB memory)
Memory: 128GB RAM
While these are not strict requirements, we recommend using a server with at least 16GB of GPU memory to successfully reproduce the experiments.

### Software Requirements
The experiments were performed on a server running `Ubuntu 20.04.1 LTS`. The following Python libraries are required to run the artifact:
```
pandas==1.5.2
pillow==10.2.0
torch==2.1.0
torchvision==0.16.0
tqdm==4.65.2
transformers==4.46.3
controlnet-aux==0.0.7
```


### Estimated Time and Storage Consumption
It is expected to take a few weeks to reproduce all the experimental results. A server with storage 1TB is suggested.

## Environment 

### Accessibility (All badges)
This github repository will be kept for a long time

### Set up the environment (Only for Functional and Reproduced badges)
Please see `src/privacy/README.md` and `src/utility/README.md` for details.

## Artifact Evaluation (Only for Functional and Reproduced badges)
Please see `src/privacy/README.md` and `src/utility/README.md` for details.

### Experiments 
Please see `src/privacy/README.md` and `src/utility/README.md` for details.

## Limitations (Only for Functional and Reproduced badges)
N/A

## Notes on Reusability (Only for Functional and Reproduced badges)
N/A
