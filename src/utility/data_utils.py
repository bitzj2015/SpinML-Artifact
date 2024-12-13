import os
import random
import pandas as pd
import collections


def synthetic_dataset(base_path_list, sampling=True):
    images = []
    states = []
    captions = []
    count = collections.defaultdict(int)
    
    for base_path in base_path_list:
        file_list = list(os.listdir(base_path))
        if sampling:
            num_samples = int(len(file_list) * base_path_list[base_path])
            select_files = random.choices(file_list, k=num_samples)
            print(f"Sample {num_samples} samples from {base_path}")
            
        else:
            select_files = file_list
            
        for filename in select_files:
            image_path = f"{base_path}/{filename}"
            state = filename.split("_")[1]
            images.append(image_path)
            states.append(state)
            captions.append(state)
            count[state] += 1
    
    df = pd.DataFrame(
        {
            'image': images, 
            'caption': captions, 
            'state': states
        }
    )
    return df