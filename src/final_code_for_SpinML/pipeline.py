import os
import random
import argparse
from label_dataset_script import label_dataset
from train_yolo_script import train_and_validate_yolo

def full_pipeline(input_folder, output_folder, prompt, label, extension, save_folder, yolo_model, device, epochs):
    """
    Execute the full pipeline: label the dataset and train YOLO models.

    Args:
        input_folder (str): Path to the input image folder.
        output_folder (str): Path to save labeled images and annotations.
        prompt (str): Prompt to find objects.
        label (str): Label for found objects.
        extension (str): File extension of images to label.
        save_folder (str): Path to save trained models and results.
        yolo_model (str): Path to the YOLO model file.
        device (str): Device to use for training (e.g., 'cuda:0').
        epochs (int): Number of training epochs.
    """
    print("Starting dataset labeling...")
    label_dataset(
        input_folder=input_folder,
        output_folder=output_folder,
        prompt=prompt,
        label=label,
        extension=extension
    )

    print("Starting YOLO model training...")
    train_and_validate_yolo(
        out_folder=output_folder,
        save_folder=save_folder,
        yolo_model=yolo_model,
        device=device,
        epochs=epochs,
        random_seed=random.randint(0, 2**20 - 1)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full pipeline: labeling and training.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input image folder.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save labeled dataset.")
    parser.add_argument("--prompt", type=str, default="pill bottle", help="Prompt to find objects.")
    parser.add_argument("--label", type=str, default="bottle", help="Label for found objects.")
    parser.add_argument("--extension", type=str, default=".png", help="Image file extension (default: .png).")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to save trained models and results.")
    parser.add_argument("--yolo_model", type=str, required=True, help="Path to the YOLO model file.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (default: 'cuda:0').")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200).")

    args = parser.parse_args()

    full_pipeline(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        prompt=args.prompt,
        label=args.label,
        extension=args.extension,
        save_folder=args.save_folder,
        yolo_model=args.yolo_model,
        device=args.device,
        epochs=args.epochs
    )
