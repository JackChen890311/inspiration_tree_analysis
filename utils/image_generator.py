import os
import argparse
import tqdm
import torch
from DiffusionUtils import DiffusionUtils

def parse_args():
    parser = argparse.ArgumentParser(description="Run diffusion generation script with arguments.")
    parser.add_argument('--dataset_name', type=str, default="v2_sub", help='Dataset name')
    parser.add_argument('--exp_name', type=str, default="20250410_0408_hybrid", help='Experiment name')
    parser.add_argument('--out_path', type=str, default="/home/jack/Code/Research/instree_analysis/experiment_image", help='Output path')
    parser.add_argument('--multiseed', action='store_true', help='Use multi-seed folder structure')
    parser.add_argument('--exp_seed', type=int, default=111, help='Experiment seed')
    parser.add_argument('--gen_seeds', type=int, nargs='+', default=[4321, 95, 11, 87654], help='List of generation seeds')
    parser.add_argument('--num_images_per_seed', type=int, default=10, help='Number of images per seed')
    parser.add_argument('--emb_name', type=str, default="learned_embeds.bin", help='Embedding name')
    return parser.parse_args()


def image_generator(args):
    exp_path = f"/home/jack/Code/Research/instree_analysis/experiments/{args.dataset_name}/{args.exp_name}"
    final_out_path = f"{args.out_path}/{args.dataset_name}/{args.exp_name}"
    if os.path.exists(final_out_path):
        print(f"Output path {final_out_path} already exists. Please remove it before running the script.")
        return
    
    for cpt_name in tqdm.tqdm(os.listdir(exp_path + "/outputs")):
        if not os.path.isdir(exp_path + "/outputs/" + cpt_name):
            continue
        if args.multiseed:
            concept_path = f"{exp_path}/outputs/{cpt_name}/v0/{args.emb_name}"
            if not os.path.exists(concept_path):
                for seed in [0, 111, 1000, 1234]:
                    concept_path = f"{exp_path}/outputs/{cpt_name}/v0/v0_seed{seed}/embeds/learned_embeds-steps-1000.bin"
                    if os.path.exists(concept_path):
                        concept_path = f"{exp_path}/outputs/{cpt_name}/v0/v0_seed{seed}/{args.emb_name}"
                        break
        else:
            concept_path = f"{exp_path}/outputs/{cpt_name}/v0/v0_seed{args.exp_seed}/{args.emb_name}"
        
        concepts = torch.load(concept_path)
        DiffusionUtils.reset_vocab()
        DiffusionUtils.add_new_vocab('<*>', concepts['<*>'])
        DiffusionUtils.add_new_vocab('<&>', concepts['<&>'])
        print(f"Generating images for {cpt_name} at {final_out_path}/{cpt_name}")

        for i, seed in enumerate(args.gen_seeds):
            DiffusionUtils.run_prompt(
                "<*>", args.num_images_per_seed,
                f"{final_out_path}/{cpt_name}/v1",
                seed, i * args.num_images_per_seed
            )
            DiffusionUtils.run_prompt(
                "<&>", args.num_images_per_seed,
                f"{final_out_path}/{cpt_name}/v2",
                seed, i * args.num_images_per_seed
            )

if __name__ == "__main__":
    args = parse_args()
    image_generator(args)
