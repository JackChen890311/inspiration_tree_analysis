import os
import torch
import matplotlib.pyplot as plt
from typing import List

class ScoreKeeper:
    def __init__(self, base_path, concept, node, seed):
        self.base_path = base_path
        self.concept = concept
        self.node = node
        self.seed = seed
        self.final_path = os.path.join(self.base_path, self.concept, self.node, f"{self.node}_seed{self.seed}/consistency_test")
        self.consistency_score = torch.load(os.path.join(self.final_path, f"seed{self.seed}_scores.bin"))
        self.steps = sorted(list(self.consistency_score.keys()))
        self.calculate_scores()
    
    def calculate_scores(self):
        self.final_score = [self.consistency_score[step]['final'] for step in self.steps]
        self.left_score = [self.consistency_score[step]['s_l'] for step in self.steps]
        self.right_score = [self.consistency_score[step]['s_r'] for step in self.steps]
        self.in_score = [l + r for l, r in zip(self.left_score, self.right_score)]
        self.cross_score = [self.consistency_score[step]['s_lr'] for step in self.steps]
        self.mapping = {
            'final': self.final_score,
            'left': self.left_score,
            'right': self.right_score,
            'in': self.in_score,
            'cross': self.cross_score
        }

    def plot(self, types=['final'], size=(6.4, 4.8), title='Consistency Score'):
        plt.figure(figsize=size)
        plt.title(title)
        for type in types:
            plt.plot(self.steps, self.mapping[type], label=type)
        plt.xlabel('Step')
        plt.ylabel('Consistency Score')
        plt.legend()
        plt.show()

    def __str__(self):
        self.plot(size=(3, 2), title=f"{self.base_path}/{self.concept}/{self.node}/seed_{self.seed}")
        return f"""
Concept: {self.concept}
Base Path: {self.base_path}
Node: {self.node}
Seed: {self.seed}
Consistency Score (Final): {self.final_score}
"""


def load_score_keepers(path, node) -> List[ScoreKeeper]:
    score_keeper_dict = {}
    for concept in os.listdir(path):
        if not os.path.isdir(os.path.join(path, concept)):
            continue
        score_keeper = None
        for seed in os.listdir(os.path.join(path, concept, node)):
            try:
                seed = int(seed.split('_')[-1][4:])
                score_keeper = ScoreKeeper(path, concept, node, seed)
                break
            except Exception as e:
                continue
        score_keeper_dict[concept] = score_keeper
    return score_keeper_dict


def load_exp(dataset_name, exp_date_name, node_name = "v0"):
    """
    Load the score keepers for a given experiment date and dataset name.
    
    Args:
        dataset_name (str): The name of the dataset.
        exp_date_name (str): The date and name of the experiment.
        node_name (str): The name of the node. Default is "v0".
    
    Returns:
        list: A list of score keepers.
    """
    exp_path = f"experiments/{dataset_name}/{exp_date_name}/outputs"
    sk_list = load_score_keepers(exp_path, node_name)
    return sk_list