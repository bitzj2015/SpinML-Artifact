import os
import random
import argparse
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

def label_dataset(input_folder, output_folder, prompt, label, extension=".png"):
    """
    Labels images in a folder using the GroundedSAM model.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to save labeled images and annotations.
        prompt (str): Prompt used to find objects.
        label (str): Label assigned to found objects.
        extension (str): File extension of images to label (default: .png).
    """
    base_model = GroundedSAM(
        ontology=CaptionOntology({prompt: label})
    )

    print(f"Labeling images in: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)
    base_model.label(
        input_folder=input_folder,
        extension=extension,
        output_folder=output_folder,
    )
    print(f"Saved labeled dataset to: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label images in a dataset using GroundedSAM.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input image folder.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save labeled dataset.")
    parser.add_argument("--prompt", type=str, default="pill bottle", help="Prompt to find objects.")
    parser.add_argument("--label", type=str, default="bottle", help="Label for found objects.")
    parser.add_argument("--extension", type=str, default=".png", help="Image file extension (default: .png).")

    args = parser.parse_args()

    label_dataset(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        prompt=args.prompt,
        label=args.label,
        extension=args.extension,
    )