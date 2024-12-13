import os
import json
import random
import argparse
from ultralytics import YOLO

def train_and_validate_yolo(out_folder, save_folder, yolo_model, device, epochs, random_seed):
    """
    Train and validate YOLO models on a given dataset.

    Args:
        out_folder (str): Folder containing the dataset.
        save_folder (str): Folder to save the trained model and results.
        yolo_model (str): Path to the YOLO model.
        device (str): Device to run the training (e.g., 'cuda:0').
        epochs (int): Number of training epochs.
        random_seed (int): Random seed for reproducibility.
    """
    data = {"model_name": out_folder, "seed": random_seed}
    accuracy = {}

    os.makedirs(save_folder, exist_ok=True)

    if out_folder not in os.listdir(save_folder):
        yolo = YOLO(yolo_model)

        yolo.train(
            data=f"{out_folder}/data.yaml", 
            project=save_folder, 
            name=out_folder, 
            device=device, 
            epochs=epochs, 
            patience=10, 
            seed=random_seed
        )
    else:
        yolo = YOLO(f"{save_folder}/{out_folder}/weights/best.pt")

    # Validate the model on server-side dataset
    print("Validation of the server-side validation dataset...")
    metrics = yolo.val(data=f"{out_folder}/data.yaml", save_json=True, device=device)

    accuracy["mAP50-95"] = metrics.box.map
    accuracy["mAP50"] = metrics.box.map50
    accuracy["mAP75"] = metrics.box.map75

    data["validation_accuracy"] = accuracy.copy()
    accuracy = {}

    with open(f"{save_folder}/val_server_results.json", "a+") as fd:
        json.dump(data, fd)
        fd.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate YOLO models.")
    parser.add_argument("--out_folder", type=str, required=True, help="Path to the dataset folder.")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to save trained models and results.")
    parser.add_argument("--yolo_model", type=str, required=True, help="Path to the YOLO model file.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (default: 'cuda:0').")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200).")
    parser.add_argument("--seed", type=int, default=random.randint(0, 2**20 - 1), help="Random seed for reproducibility.")

    args = parser.parse_args()

    train_and_validate_yolo(
        out_folder=args.out_folder,
        save_folder=args.save_folder,
        yolo_model=args.yolo_model,
        device=args.device,
        epochs=args.epochs,
        random_seed=args.seed,
    )