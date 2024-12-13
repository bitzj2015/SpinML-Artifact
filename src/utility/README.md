# 

# Required libraries
To run the scripts, ensure the following libraries are installed:
```
pandas=1.5.2
pillow=10.2.0
torch=2.1.0
torchvision=0.16.0
tqdm=4.65.2
```

# Run commands
```
python train_mobilenet.py --alpha $alpha --random_seed $seed --testdata $testdata --augdata $augdata
```
Example:
```
./train_mobilenet_example.sh
```