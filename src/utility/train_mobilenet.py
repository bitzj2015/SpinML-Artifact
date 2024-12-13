from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import DataLoader
from train_utils import *
from data_utils import *
import torch
import argparse
import random

# Set up the argument parser
parser = argparse.ArgumentParser(description='Process input arguments')
parser.add_argument('--alpha', type=float, default=0.8, required=True, help='Alpha value for dataset weighting')
parser.add_argument('--random_seed', type=int, default=100, required=True, help='Random seed for reproducibility')
parser.add_argument('--eval', action='store_true', help='Evaluation mode')
parser.add_argument('--testdata', type=str, default="husky", required=False, help='Test dataset')
parser.add_argument('--augdata', type=str, default="L1_L0", required=False, help='Test dataset')

# Parse arguments
args = parser.parse_args()

# Use the parsed arguments
alpha = args.alpha
random_seed = args.random_seed
eval_only = args.eval
testdata = args.testdata
augdata_version = args.augdata

random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

folder_path = './saved_models'
os.makedirs(folder_path, exist_ok=True)

SYNTHETIC_DATASET_PATH = {
    f"../../data/husky/synthetic/L0_L0/clean": alpha,
    f"../../data/husky/synthetic/{augdata_version}/clean": 1 - alpha
}

DATASET = f"husky_syn_{augdata_version}_{random_seed}_{alpha}"

# Assuming you have a DataFrame named 'df' with 'image_path' and 'caption' columns
# You can define the transformations as needed
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the training dataset
train_df = synthetic_dataset(base_path_list=SYNTHETIC_DATASET_PATH)
train_dataset = ImageCaptionDataset(train_df, transform=transform)

# Define the sizes of your training and validation sets
train_size = int(0.9 * len(train_dataset))  # 90% for training
val_size = len(train_dataset) - train_size  # Remaining 10% for testing

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create the testing dataset
test_df = synthetic_dataset(
    base_path_list={
        f"../../data/husky/real/raw": 1.0
    },
    sampling=False
)
test_dataset = ImageCaptionDataset(test_df, transform=transform)


# Create separate data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


if not eval_only:
    # Initialize the MobileNetCaptioning model
    model = MobileNetCaptioning(num_classes=4)

    # Set hyperparameters
    num_epochs = 5 # Adjust as needed
    learning_rate = 0.001  # Adjust as needed

    # Train the model
    train(model, train_loader, num_epochs, learning_rate, val_loader, DATASET)

# Evaluate the model on the testing set


model = torch.load(f'./saved_models/mobilenet_{DATASET}.pth')
test_accuracy = evaluate(model, test_loader)
print("Testing accuracy", test_accuracy)
