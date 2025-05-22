import os


def list_exp_names(dataset_name, base_path="experiments"):
    """
    List all experiment names for a given dataset.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        list: A list of experiment names.
    """
    exp_names = sorted(os.listdir(f"{base_path}/{dataset_name}"))
    if ".DS_Store" in exp_names:
        exp_names.remove(".DS_Store")
    print(f"===== Experiment names for {dataset_name}: =====")
    for i, name in enumerate(exp_names):
        print(i, name)
    print(f"===== Total experiments: {len(exp_names)} =====")
    return exp_names


def list_concept_names(dataset_name, base_path="experiment_data"):
    """
    List all concept names for a given dataset.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        list: A list of concept names.
    """
    cpt_names = sorted(os.listdir(f"{base_path}/{dataset_name}"))
    if ".DS_Store" in cpt_names:
        cpt_names.remove(".DS_Store")
    print(f"===== Concept names for {dataset_name}: =====")
    for name in cpt_names:
        print(name)
    print(f"===== Total concepts: {len(cpt_names)} =====")
    return cpt_names
