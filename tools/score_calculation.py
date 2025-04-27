import os
import torch
import argparse
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, AutoImageProcessor, AutoModel
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", type=str, required=True, help="Path to the first image folder")
    parser.add_argument("--folder2", type=str, required=True, help="Path to the second image folder")
    parser.add_argument("--model_type", type=str, choices=["clip", "dino"], required=True, help="Choose 'clip' or 'dino'")
    parser.add_argument("--model_id_clip", type=str, default="openai/clip-vit-base-patch32", help="CLIP model ID") # openai/clip-vit-large-patch14 for clip large
    parser.add_argument("--model_id_dino", type=str, default="facebook/dino-vits16", help="DINO model ID") # facebook/dinov2-base for v2
    parser.add_argument("--show_sim_detail", action="store_true", help="Show detailed similarity matrix")
    parser.add_argument("--get_sim_detail", action="store_true", help="Get detailed similarity matrix")
    return parser.parse_args()


def load_images_with_names(folder, preprocess, device):
    images = []
    names = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            path = os.path.join(folder, filename)
            image = Image.open(path).convert('RGB')
            tensor = preprocess(image, return_tensors="pt")["pixel_values"].squeeze(0)
            images.append(tensor)
            names.append(filename)
    return torch.stack(images).to(device), names


def compute_similarity(embeds_a, embeds_b):
    embeds_a = embeds_a / embeds_a.norm(dim=1, keepdim=True)
    embeds_b = embeds_b / embeds_b.norm(dim=1, keepdim=True)
    return (embeds_a @ embeds_b.T).cpu().numpy()


def score_calculation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_type == "clip":
        model = CLIPModel.from_pretrained(args.model_id_clip).to(device)
        preprocess = CLIPImageProcessor.from_pretrained(args.model_id_clip)

    elif args.model_type == "dino":
        model = AutoModel.from_pretrained(args.model_id_dino).to(device)
        preprocess = AutoImageProcessor.from_pretrained(args.model_id_dino)

    else:
        raise ValueError("Invalid model type. Choose 'clip' or 'dino' — don’t make me babysit your typos.")

    images_a, names_a = load_images_with_names(args.folder1, preprocess, device)
    images_b, names_b = load_images_with_names(args.folder2, preprocess, device)

    with torch.no_grad():
        if args.model_type == "clip":
            features_a = model.get_image_features(images_a)
            features_b = model.get_image_features(images_b)
        else:  # DINO case
            features_a = model(images_a).last_hidden_state.mean(dim=1)
            features_b = model(images_b).last_hidden_state.mean(dim=1)

        features_a = features_a / features_a.norm(dim=1, keepdim=True)
        features_b = features_b / features_b.norm(dim=1, keepdim=True)

    similarity_matrix = compute_similarity(features_a, features_b)

    # 排除檔名相同的情況
    for i, name_a in enumerate(names_a):
        for j, name_b in enumerate(names_b):
            if name_a == name_b:
                similarity_matrix[i, j] = np.nan  # 標記為NaN，表示不計算自己比自己

    if args.show_sim_detail:
        print("Similarity Matrix (NaN = skipped same-name pairs):")
        print("Rows: images from folder1")
        print("Columns: images from folder2")
        print(names_a)
        print(names_b)
        print("Similarity Matrix:")
        print(similarity_matrix)

    # 計算有效值的平均相似度
    valid_scores = similarity_matrix[~np.isnan(similarity_matrix)]
    avg_similarity = valid_scores.mean() if valid_scores.size > 0 else 0.0
    print(f"Average Similarity: {avg_similarity:.4f}")

    return (avg_similarity, similarity_matrix, (names_a, names_b)) if args.get_sim_detail else avg_similarity


if __name__ == "__main__":
    args = parse_args()
    avg_similarity = score_calculation(args)