#!/bin/bash

# Define the arrays of values for alpha, random_seed, and testdata
alphas=(0.0)
seeds=(1000)
testdata="husky"
augdatas=("L2_L0")

echo $augdata

# Loop over each alpha
for alpha in "${alphas[@]}"; do
    # Loop over each random_seed
    for seed in "${seeds[@]}"; do
        # Loop over each testdata
        for augdata in "${augdatas[@]}"; do
            # Construct and run the command
            echo ">>>>>>>>>>>>>>>>> augdata=$augdata, alpha=$alpha, seed=$seed, testdata=$testdata >>>>>>>>>>>>>>>>>"
            python train_mobilenet.py --alpha $alpha --random_seed $seed --testdata $testdata --augdata $augdata
        done
    done
done
echo "All experiments completed."