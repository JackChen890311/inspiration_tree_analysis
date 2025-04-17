import numpy as np
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

def plot_one_exp(score_keeper_dict, types=['final'], size=(10, 5), title='Consistency Score Comparison'):
    score_keeper_list = list(score_keeper_dict.values())
    plt.figure(figsize=size)
    plt.title(title)
    for score_keeper in score_keeper_list:
        for type in types:
            plt.plot(score_keeper.steps, score_keeper.mapping[type], label=f"{score_keeper.concept}/{score_keeper.node}/seed_{score_keeper.seed}/{type}")
    plt.xlabel('Step')
    plt.ylabel('Consistency Score')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

def plot_exp_comparison(list_of_score_keeper_dict, exp_name, types=['final'], size=(10, 5), title='Consistency Score Comparison'):
    plt.figure(figsize=size)
    plt.title(title)
    for type in types:
        for i, score_keeper_dict in enumerate(list_of_score_keeper_dict):
            average = get_average(score_keeper_dict, type)
            plt.plot(score_keeper_dict[list(score_keeper_dict.keys())[0]].steps, average, label=f"{exp_name[i]}/{type}")
    plt.xlabel('Step')
    plt.ylabel('Consistency Score')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

def plot_concept_comparison(dict_list, label_list, field='final', title='Concept Score Comparison'):
    title += " ({})".format(field)
    assert len(dict_list) == len(label_list), "The number of dictionaries and labels must match."
    
    # Get common concepts across all dictionaries
    concept_sets = [set(d.keys()) for d in dict_list]
    common_concepts = sorted(set.intersection(*concept_sets))
    
    if not common_concepts:
        print("No common concepts to compare.")
        return

    n_dicts = len(dict_list)
    x = np.arange(len(common_concepts))
    width = 0.8 / n_dicts  # Adjust total bar width

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (d, label) in enumerate(zip(dict_list, label_list)):
        scores = [d[c].mapping[field][-1] for c in common_concepts]
        bars = ax.bar(x + i * width - (n_dicts - 1) * width / 2, scores, width, label=label)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_xlabel('Concept')
    ax.set_ylabel('Final Score (Last Step)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(common_concepts, rotation=45, ha='right')
    ax.set_ylim(0, 2.5)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def summary_exp(sk_list, exp_name):
    """
    Print a summary of the experiment.
    Params:
        sk_list (list): List of score keepers.
        exp_name (str): Name of the experiment.
    """
    print(f"Summary for {exp_name}:")
    print(sk_list.keys())
    print(get_average_final(sk_list, type='final'))
    print(get_variance_final(sk_list, type='final'))

def get_concept_score(list_of_sk_list, concept_name):
    for sk_list in list_of_sk_list:
        print(sk_list[concept_name])
