from transformers import Blip2Model, AutoProcessor
import torch
import os
from PIL import Image


def process_images_batch(image_path, model, processor, batch_size):
    all_features = []
    for i in range(0, len(image_path), batch_size):
        batch_urls = image_path[i:i+batch_size]
        images = [Image.open(url) for url in batch_urls]
        inputs = processor(images=images, return_tensors="pt", padding=True).to("cuda", torch.float16)

        with torch.no_grad():
            batch_features = model.get_image_features(**inputs)
            all_features.append(batch_features.pooler_output)
    
    # Concatenate all batch features into a single tensor
    return torch.cat(all_features, dim=0)

def get_embedding(image_path, embeddings_file, batch_size=20):
    if os.path.exists(embeddings_file):
        embeddings_matrix = torch.load(embeddings_file)

    else:
        model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b").cuda()
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        embeddings_matrix = process_images_batch(image_path, model, processor, batch_size)
        torch.save(embeddings_matrix, embeddings_file)

    return embeddings_matrix

# Example usage
average = []
data = {
    "husky_L0_L0": [
        "../../data/husky/real/raw_split",
        "../../data/husky/synthetic/L0_L0/clean_split"
    ],
    "husky_L1_L0": [
        "../../data/husky/real/raw_split",
        "../../data/husky/synthetic/L1_L0/clean_split"
    ],
    "husky_L2_L0": [
        "../../data/husky/real/raw_split",
        "../../data/husky/synthetic/L2_L0/clean_split"
    ],
}


for key in data.keys():
    for object_type in ["foreground", "background"]:
        image_raw_dir = data[key][0] + f"/{object_type}"
        image_pred_dir = data[key][1] + f"/{object_type}"
        emb_raw_file = f"{image_raw_dir}_embeddings.pkl"
        emb_pred_file = f"{image_pred_dir}_embeddings.pkl"

        image_raw_path = [os.path.join(image_raw_dir, filename) for filename in os.listdir(image_raw_dir)]
        image_pred_path = [os.path.join(image_pred_dir, filename) for filename in os.listdir(image_pred_dir)]


        emb_raw = get_embedding(image_raw_path, emb_raw_file)
        emb_pred = get_embedding(image_pred_path, emb_pred_file)
        cos_mat = torch.matmul(emb_pred, emb_raw.t()) / torch.norm(emb_pred, dim=-1).reshape(-1,1) / torch.norm(emb_raw, dim=-1).reshape(1,-1)
        print(f"Sanitizer: {key}, Object: {object_type}, SIM: {cos_mat.mean().item()}")


