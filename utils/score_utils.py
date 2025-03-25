import os
import matplotlib.pyplot as plt

def get_average(score_keeper_dict, type):
    score_keeper_list = list(score_keeper_dict.values())
    steps = score_keeper_list[0].steps
    average = [0 for _ in range(len(steps))]
    for score_keeper in score_keeper_list:
        for i, step in enumerate(steps):
            average[i] += score_keeper.mapping[type][i]
    average = [score / len(score_keeper_list) for score in average]
    return average 

def get_average_final(score_keeper_dict, type):
    return get_average(score_keeper_dict, type)[-1]

def get_variance(score_keeper_dict, type):
    score_keeper_list = list(score_keeper_dict.values())
    n = len(score_keeper_list)
    if n <= 1:
        return [0 for _ in range(len(score_keeper_list[0].steps))] if n == 1 else []
    steps = score_keeper_list[0].steps
    variance = [0 for _ in range(len(steps))]
    average = get_average(score_keeper_dict, type)
    for score_keeper in score_keeper_list:
        for i, step in enumerate(steps):
            variance[i] += (score_keeper.mapping[type][i] - average[i]) ** 2
    return [score / (n - 1) for score in variance]
    
def get_variance_final(score_keeper_dict, type):
    return get_variance(score_keeper_dict, type)[-1]

def plot_one_exp(score_keeper_dict, types=['final'], size=(6.4, 4.8), title='Consistency Score Comparison'):
    score_keeper_list = list(score_keeper_dict.values())
    plt.figure(figsize=size)
    plt.title(title)
    for score_keeper in score_keeper_list:
        for type in types:
            plt.plot(score_keeper.steps, score_keeper.mapping[type], label=f"{score_keeper.concept}/{score_keeper.node}/seed_{score_keeper.seed}/{type}")
    plt.xlabel('Step')
    plt.ylabel('Consistency Score')
    plt.legend()
    plt.show()

def plot_exp_comparison(list_of_score_keeper_dict, exp_name, types=['final'], size=(6.4, 4.8), title='Consistency Score Comparison'):
    plt.figure(figsize=size)
    plt.title(title)
    for type in types:
        for i, score_keeper_dict in enumerate(list_of_score_keeper_dict):
            average = get_average(score_keeper_dict, type)
            plt.plot(score_keeper_dict[list(score_keeper_dict.keys())[0]].steps, average, label=f"{exp_name[i]}/{type}")
    plt.xlabel('Step')
    plt.ylabel('Consistency Score')
    plt.legend()
    plt.show()