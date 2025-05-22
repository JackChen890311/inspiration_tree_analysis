import os
import argparse
import pickle as pk
from score_calculation import score_calculation

class ARGS:
    folder1 = ""
    folder2 = ""
    model_type = ""
    model_id_clip = "openai/clip-vit-base-patch32"
    model_id_dino = "facebook/dino-vits16"  # dinov2-base for v2
    show_sim_detail = False  # Set to True to show detailed similarity matrix
    get_sim_detail = False  # Set to True to get detailed similarity matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate experiments")
    parser.add_argument("--exp_img_dir", type=str, required=True)
    parser.add_argument("--origin_img_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="clip", choices=["clip", "dino"])
    parser.add_argument("--out_sub_name", type=str, default="")
    return parser.parse_args()


def evaluate_experiments(args):
    exp_img_dir = args.exp_img_dir
    origin_img_dir = args.origin_img_dir
    output_dir = args.output_dir
    ARGS.model_type = args.model_type

    result = {}
    for cpt_name in os.listdir(exp_img_dir):
        print(f"Evaluating {cpt_name}...")
        ARGS.folder1 = os.path.join(exp_img_dir, cpt_name, "v1")
        ARGS.folder2 = os.path.join(exp_img_dir, cpt_name, "v1")
        v1v1 = score_calculation(ARGS)
        ARGS.folder1 = os.path.join(exp_img_dir, cpt_name, "v2")
        ARGS.folder2 = os.path.join(exp_img_dir, cpt_name, "v2")
        v2v2 = score_calculation(ARGS)

        ARGS.folder1 = os.path.join(exp_img_dir, cpt_name, "v1")
        ARGS.folder2 = os.path.join(exp_img_dir, cpt_name, "v2")
        v1v2 = score_calculation(ARGS)

        ARGS.folder1 = os.path.join(origin_img_dir, cpt_name, "v0")
        ARGS.folder2 = os.path.join(exp_img_dir, cpt_name, "v1")
        v0v1 = score_calculation(ARGS)
        ARGS.folder1 = os.path.join(origin_img_dir, cpt_name, "v0")
        ARGS.folder2 = os.path.join(exp_img_dir, cpt_name, "v2")
        v0v2 = score_calculation(ARGS)
        
        ARGS.folder1 = os.path.join(origin_img_dir, cpt_name, "v0")
        ARGS.folder2 = os.path.join(origin_img_dir, cpt_name, "v0")
        v0v0 = score_calculation(ARGS)

        consistency_score = v1v1 + v2v2
        distinction_score = v1v2
        relevance_score = v0v1 + v0v2

        result[cpt_name] = {
            "consistency_score": consistency_score,
            "distinction_score": distinction_score,
            "relevance_score": relevance_score,
            "v0v0": v0v0,
            "v1v1": v1v1,
            "v2v2": v2v2,
            "v0v1": v0v1,
            "v0v2": v0v2,
            "v1v2": v1v2,
        }
        print(f"Results for {cpt_name}: {result[cpt_name]}")

    # Save results to output directory
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{args.exp_img_dir.split('/')[-1]}_{ARGS.model_type}{args.out_sub_name}.pkl"), "wb") as f:
        pk.dump(result, f)
    return result

if __name__ == "__main__":
    args = parse_args()
    result = evaluate_experiments(args)