from Testing.DataLoader import DataLoader
from itertools import product


class ParameterCreator:
    def __init__(self):
        self.parameters = {}
        data_loader = DataLoader(1, False, 23)
        train_df, test_df = data_loader.create_train_test_df(True, True, True)

        vals = {
            'min_cluster_size': [5],
            'min_samples': [5],
            'n_neighbors': [int(train_df.shape[0] - 2)],
            'min_dist': [0],
            'num_components': [2],
            'no_umap': [False],
            'parametric_umap': [False],
            'claim_column_name': ['Text'],
            'veracity_column_name': ['Numerical Rating'],
            'supervised_label_column_name': ['Numerical Rating'],
            'random_seed': [True],
            'use_weightage': [True],
            'k': [15000],
            'random_seed_val': [23],
            'threshold_break': [0.9],
            'break_further': [True],
            'use_hdbscan': [True],
            'supervised_umap': [True],
        }

        # Generate all combinations of parameter values
        keys, values = zip(*vals.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]

        # Store all parameter combinations
        self.parameters = param_combinations

    def get_parameters(self):
        return self.parameters

