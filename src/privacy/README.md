# Privacy Measurement for Customized Synthetic Data Generation
This folder contains scripts to measure the privacy of customized synthetic data generation methods.

# Required libraries
To run the scripts, ensure the following libraries are installed:
```
transformers=4.46.3
pillow=10.2.0
torch=2.1.0
controlnet-aux=0.0.7
```

Install them using `pip` if needed:
```
pip install transformers==4.46.3 pillow==10.2.0 torch==2.1.0 controlnet-aux==0.0.7
```

# Run commands
1. To measure the mutual information (MI) between sanitized image segments and raw image segments, run:
```
python sim_mi.py
```
Example output:
```
Sanitizer: husky_L1_L0, Object: foreground, MI: 0.008926574318232247
Sanitizer: husky_L2_L0, Object: foreground, MI: 0.39831607422738874
Sanitizer: husky_L1_L0, Object: background, MI: 0.007292497667598251
Sanitizer: husky_L2_L0, Object: background, MI: 0.11319264028298671
```
2. To measure the similarity (SIM) between synthetic image segments and raw image segments, run:
```
python sim_img2img2.py
```
Example output:
```
Sanitizer: husky_L0_L0, Object: foreground, SIM: 0.6229013800621033
Sanitizer: husky_L0_L0, Object: background, SIM: 0.5519641637802124
Sanitizer: husky_L1_L0, Object: foreground, SIM: 0.7152029871940613
Sanitizer: husky_L1_L0, Object: background, SIM: 0.5705515742301941
Sanitizer: husky_L2_L0, Object: foreground, SIM: 0.7166070938110352
Sanitizer: husky_L2_L0, Object: background, SIM: 0.572185218334198
```

