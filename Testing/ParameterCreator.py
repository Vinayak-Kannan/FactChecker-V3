from Testing.DataLoader import DataLoader
from itertools import product


class ParameterCreator:
    def __init__(self):
        self.parameters = {}
        data_loader = DataLoader(0.75, False, 23)
        train_df, test_df = data_loader.create_train_test_df(True, True, True, True)

        vals = {
            # HDBSCAN parameters
            'min_cluster_size': [5],
            'min_samples': [2],
            'use_hdbscan': [True],

            # UMAP parameters
            'n_neighbors': [int(train_df.shape[0] - 2)],
            'min_dist': [0],
            'num_components': [100],

            # UMAP options
            'no_umap': [False],
            'parametric_umap': [True],
            'supervised_umap': [False],

            # Data column specifications
            'claim_column_name': ['Text'],
            'veracity_column_name': ['Numerical Rating'],
            'supervised_label_column_name': ['Numerical Rating'],

            # Random seed options
            'random_seed': [True],
            'random_seed_val': [23],

            # Other pipeline options
            'use_weightage': [True],
            'k': [15000],
            'threshold_break': [0.9],
            'break_further': [True],
            'size_of_dataset': [1],
            'use_only_CARD': [True]
        }

        # Generate all combinations of parameter values
        keys, values = zip(*vals.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]

        # Store all parameter combinations
        self.parameters = param_combinations

    def get_parameters(self):
        return self.parameters
 
