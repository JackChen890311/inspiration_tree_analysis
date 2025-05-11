import os
import argparse
import tqdm
import torch
from DiffusionUtils import DiffusionUtils
from exp_utils import list_exp_names, list_concept_names

def parse_args():
    parser = argparse.ArgumentParser(description="Run diffusion generation script with arguments.")
    parser.add_argument('--dataset_name', type=str, default="v2_sub", help='Dataset name')
    parser.add_argument('--exp_name', type=str, default="20250410_0408_hybrid", help='Experiment name')
    parser.add_argument('--out_path', type=str, default="/home/jack/Code/Research/instree_analysis/experiment_image", help='Output path')
    parser.add_argument('--multiseed', action='store_true', help='Use multi-seed folder structure')
    parser.add_argument('--exp_seed', type=int, default=111, help='Experiment seed')
    parser.add_argument('--gen_seeds', type=int, nargs='+', default=[4321, 95, 11, 87654], help='List of generation seeds')
    parser.add_argument('--num_images_per_seed', type=int, default=10, help='Number of images per seed')
    return parser.parse_args()

def image_generator(args):
    exp_path = f"/home/jack/Code/Research/instree_analysis/experiments/{args.dataset_name}/{args.exp_name}"
    
    for cpt_name in tqdm.tqdm(os.listdir(exp_path + "/outputs")):
        if not os.path.isdir(exp_path + "/outputs/" + cpt_name):
            continue
        if args.multiseed:
            concept_path = f"{exp_path}/outputs/{cpt_name}/v0/learned_embeds.bin"
        else:
            concept_path = f"{exp_path}/outputs/{cpt_name}/v0/v0_seed{args.exp_seed}/learned_embeds.bin"
        
        concepts = torch.load(concept_path)
        DiffusionUtils.reset_vocab()
        DiffusionUtils.add_new_vocab('<*>', concepts['<*>'])
        DiffusionUtils.add_new_vocab('<&>', concepts['<&>'])

        for i, seed in enumerate(args.gen_seeds):
            DiffusionUtils.run_prompt(
                "<*>", args.num_images_per_seed,
                f"{args.out_path}/{args.dataset_name}/{args.exp_name}/{cpt_name}/v1",
                seed, i * args.num_images_per_seed
            )
            DiffusionUtils.run_prompt(
                "<&>", args.num_images_per_seed,
                f"{args.out_path}/{args.dataset_name}/{args.exp_name}/{cpt_name}/v2",
                seed, i * args.num_images_per_seed
            )

if __name__ == "__main__":
    args = parse_args()
    image_generator(args)
