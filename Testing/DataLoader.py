import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class DataLoader():
    def __init__(self, percentage_false: float, verbose: bool = False, random_seed: int = 23):
        self.verbose = verbose
        np.random.seed(seed=random_seed)
        self.percentage_false = percentage_false
        # Ground Truth
        self.ground_truth = pd.read_csv(
            "../../Clustering/Raw Data/Climate/VKs Copy of Cleaned - Google Fact Check Explorer - Climate.xlsx - Corrected.csv")
        self.ground_truth = self.ground_truth.dropna(subset=['Text'])
        self.ground_truth = self.ground_truth[self.ground_truth['Text'] != '']
        self.ground_truth = self.ground_truth.drop_duplicates(subset=['Text'])
        self.ground_truth['Synthetic'] = [False for i in range(len(self.ground_truth))]

        # EPA
        self.epa_who_data = pd.read_csv(
            "../../Clustering/Raw Data/Climate/climate_change_epa_who.csv")
        self.epa_who_data['Category'] = -1
        self.epa_who_data['Numerical Rating'] = 3
        self.epa_who_data = self.epa_who_data.dropna(subset=['Text'])
        self.epa_who_data = self.epa_who_data[self.epa_who_data['Text'] != '']
        self.epa_who_data = self.epa_who_data.drop_duplicates(subset=['Text'])
        self.epa_who_data['Synthetic'] = [False for i in range(len(self.epa_who_data))]


        self.epa_who_data_negated = pd.read_csv(
            "../../Clustering/Raw Data/Climate/Negated Claims/negated_epa_data.csv")
        self.epa_who_data_negated['Category'] = -1
        self.epa_who_data_negated['Numerical Rating'] = 1
        self.epa_who_data_negated = self.epa_who_data_negated.rename(columns={'text': 'Text'})
        self.epa_who_data_negated = self.epa_who_data_negated.dropna(subset=['Text'])
        self.epa_who_data_negated = self.epa_who_data_negated[self.epa_who_data_negated['Text'] != '']
        self.epa_who_data_negated = self.epa_who_data_negated.drop_duplicates(subset=['Text'])
        self.epa_who_data_negated['Synthetic'] = [True for i in range(len(self.epa_who_data_negated))]

        # Card Data
        self.card_data = pd.read_csv(
            "../../Clustering/Raw Data/Climate/card_train_with_score.csv")
        self.card_data['claim'] = '1_1'
        self.card_data = self.card_data.rename(columns={'text': 'Text', 'claim': 'Category'})
        # To the card_data, add a 'Numerical Rating' column with value 1
        self.card_data['Numerical Rating'] = 1
        self.card_data = self.card_data.dropna(subset=['Text'])
        self.card_data = self.card_data[self.card_data['Text'] != '']
        self.card_data = self.card_data[self.card_data['score'] >= 0.8]
        self.card_data = self.card_data.drop_duplicates(subset=['Text'])
        self.card_data['Synthetic'] = [False for i in range(len(self.card_data))]


        self.card_data_negated = pd.read_csv(
            "../../Clustering/Raw Data/Climate/Negated Claims/negated_card_data_score_over_0.7.csv")
        self.card_data_negated['claim'] = '1_1'
        self.card_data_negated = self.card_data_negated.rename(columns={'text': 'Text', 'claim': 'Category'})
        # To the card_data, add a 'Numerical Rating' column with value 1
        self.card_data_negated['Numerical Rating'] = 3
        # Drop the Unnamed: 0 column
        self.card_data_negated = self.card_data_negated.drop(columns=['Unnamed: 0'])
        self.card_data_negated = self.card_data_negated.dropna(subset=['Text'])
        self.card_data_negated = self.card_data_negated[self.card_data_negated['Text'] != '']
        self.card_data_negated = self.card_data_negated[self.card_data_negated['score'] >= 0.8]
        self.card_data_negated = self.card_data_negated.drop_duplicates(subset=['Text'])
        self.card_data_negated['Synthetic'] = [True for i in range(len(self.card_data_negated))]
        
	# FEVER
        self.FEVER_data = pd.read_csv("../../LLMTesting/fever_data.csv")
    def create_train_test_df(self, use_card_data: bool, use_epa_data: bool, use_ground_truth: bool, use_fever: bool, percentage_data_to_use: float = 1.0) -> (
    pd.DataFrame, pd.DataFrame):
        self.ground_truth = self.ground_truth[self.ground_truth['Numerical Rating'].isin([1, 3])]

        # Drop all columns except 'Text' and 'Numerical Rating'
        self.ground_truth = self.ground_truth[['Text', 'Numerical Rating']]
        objects = []
        if use_ground_truth and use_epa_data and use_card_data and not use_fever:
            ground_truth_ones_count = self.ground_truth['Numerical Rating'].value_counts().get(1, 0)
            card_data_ones_count = self.card_data['Numerical Rating'].value_counts().get(1, 0)
            number_of_total_false = ground_truth_ones_count + card_data_ones_count
            number_of_true = int((1 - self.percentage_false) * number_of_total_false / self.percentage_false)
            # Randomly sample 100 rows from card_data
            sample_new = self.card_data
            sample_new['Numerical Rating'] = 1
            objects = []
            objects.append(sample_new)
            objects.append(self.ground_truth)
            # Sample percentage false number of rows for epa_who_data
            self.epa_who_data = self.epa_who_data.sample(n=min(number_of_true, len(self.epa_who_data)))
            objects.append(self.epa_who_data)
        elif use_ground_truth and use_epa_data and use_card_data and use_fever:
            print("All data sources are being used")
            # Append all dataframes
            objects.append(self.ground_truth)
            objects.append(self.card_data)
            objects.append(self.epa_who_data)
            objects.append(self.FEVER_data)
        elif use_fever:
            objects.append(self.FEVER_data)
        

        train_df = pd.concat(objects, ignore_index=True)
        train_df = train_df.drop_duplicates(subset=['Text'])
        # Check if there are any duplicates in Text
        if train_df['Text'].duplicated().any():
            raise ValueError("There are duplicates in the 'Text' column")

        train_df = train_df.sample(frac=percentage_data_to_use)
        # Select random sample of 80% such that there is an even distribution of 1s and 3s
        test_df = train_df.groupby('Numerical Rating').apply(lambda x: x.sample(frac=0.2)).reset_index(drop=True)
        # Drop the rows in test_df from train_df
        train_df = train_df[~train_df.index.isin(test_df.index)]
        # Drop duplicates in text column
        train_df = train_df.drop_duplicates(subset=['Text'])
        test_df = test_df.drop_duplicates(subset=['Text'])

        if self.verbose:
            print(train_df['Numerical Rating'].value_counts())
            print(test_df['Numerical Rating'].value_counts())

        test_data = pd.read_csv('../../Clustering/Raw Data/Climate/test_data_06_26 - Sheet1.csv')
        test_data['Numerical Rating'] = test_data['Numerical Rating'].astype(int)
        test_data['Synthetic'] = [False for i in range(len(test_data))]
        test_data = test_data.drop_duplicates(subset=['Text'])
        test_df = pd.concat([test_df, test_data], ignore_index=True)
        return train_df, test_df

    def create_large_train_test_df(self, remove_synthetic_data: bool) -> (pd.DataFrame, pd.DataFrame):
        self.ground_truth = self.ground_truth[self.ground_truth['Numerical Rating'].isin([1, 3])]
        # Randomly sample half of the rows from epa_data
        epa_data_train = self.epa_who_data.sample(frac=0.5)
        epa_data_test = self.epa_who_data[~self.epa_who_data.index.isin(epa_data_train.index)]

        epa_data_train_negated = self.epa_who_data_negated.sample(frac=0.5)
        epa_data_test_negated = self.epa_who_data_negated[~self.epa_who_data_negated.index.isin(epa_data_train_negated.index)]

        # Randomly sample half of the rows from ground_truth
        ground_truth_train = self.ground_truth.sample(frac=0.5)
        ground_truth_test = self.ground_truth[~self.ground_truth.index.isin(ground_truth_train.index)]

        # Randomly sample half of the rows from card_data
        card_data_train = self.card_data.sample(frac=0.5)
        card_data_test = self.card_data[~self.card_data.index.isin(card_data_train.index)]

        card_data_train_negated = self.card_data_negated.sample(frac=0.5)
        card_data_test_negated = self.card_data_negated[~self.card_data_negated.index.isin(card_data_train_negated.index)]

        # Concatenate the training data
        train_df = pd.concat([epa_data_train, ground_truth_train, card_data_train, epa_data_train_negated, card_data_train_negated], ignore_index=True)
        # Concatenate the testing data
        test_df = pd.concat([epa_data_test, ground_truth_test, card_data_test, epa_data_test_negated, card_data_test_negated], ignore_index=True)

        train_df['Numerical Rating'] = train_df['Numerical Rating'].astype(int)
        test_df['Numerical Rating'] = test_df['Numerical Rating'].astype(int)

        # Filter claims from train_df / test_df in the 'text' column which are in the top 10% and bottom 10% of claims in terms of length
        train_df['length'] = train_df['Text'].apply(lambda x: len(x.split()))
        test_df['length'] = test_df['Text'].apply(lambda x: len(x.split()))
        # train_df = train_df[train_df['length'] > train_df['length'].quantile(0.1)]
        # train_df = train_df[train_df['length'] < train_df['length'].quantile(0.9)]
        # test_df = test_df[test_df['length'] > test_df['length'].quantile(0.1)]
        # test_df = test_df[test_df['length'] < test_df['length'].quantile(0.9)]

        # Remove duplicates in train_df and test_df
        train_df = train_df.drop_duplicates(subset=['Text'])
        test_df = test_df.drop_duplicates(subset=['Text'])

        # Remove synthetic data
        if remove_synthetic_data:
            train_df = train_df[train_df['Synthetic'] == False]
            test_df = test_df[test_df['Synthetic'] == False]

        if self.verbose:
            print(train_df['Numerical Rating'].value_counts())
            print(test_df['Numerical Rating'].value_counts())

        return train_df, test_df

    def create_matched_large_df(self, only_use_synthetic: bool) -> (pd.DataFrame, pd.DataFrame):
        card = pd.read_csv('../../Clustering/Raw Data/Climate/Cleaned/card.csv')
        epa = pd.read_csv('../../Clustering/Raw Data/Climate/Cleaned/epa.csv')

        # Pick half of the rows from card.csv and epa.csv
        card_train = card.sample(frac=0.5)
        card_test = card[~card.index.isin(card_train.index)]
        epa_train = epa.sample(frac=0.5)
        epa_test = epa[~epa.index.isin(epa_train.index)]

        # Multiply each by self.percentage_false
        card_train = card_train.sample(frac=self.percentage_false)
        card_test = card_test.sample(frac=self.percentage_false)
        epa_train = epa_train.sample(frac=self.percentage_false)
        epa_test = epa_test.sample(frac=self.percentage_false)

        # Concatenate the training data
        train_df = pd.DataFrame(
            pd.concat([card_train['Text'], card_train['negated_text'], epa_train['Text'], epa_train['negated_text']],
                      ignore_index=True), columns=['Text'])
        synthetic_train_df_array = [False for i in range(len(card_train['Text']))] + [True for i in range(len(card_train['Text']))] + [False for i in range(len(epa_train['Text']))] + [True for i in range(len(epa_train['Text']))]
        numerical_rating = np.repeat([1, 3, 3, 1],
                                     [len(card_train['Text']), len(card_train['negated_text']),
                                      len(epa_train['Text']), len(epa_train['negated_text'])])
        train_df['Numerical Rating'] = numerical_rating
        train_df['Synthetic'] = synthetic_train_df_array

        # Concatenate the testing data
        test_df = pd.DataFrame(
            pd.concat([card_test['Text'], card_test['negated_text'], epa_test['Text'], epa_test['negated_text']],
                      ignore_index=True), columns=['Text'])
        synthetic_test_df_array = [False for i in range(len(card_test['Text']))] + [True for i in range(len(card_test['Text']))] + [False for i in range(len(epa_test['Text']))] + [True for i in range(len(epa_test['Text']))]
        numerical_rating = np.repeat([1, 3, 3, 1],
                                     [len(card_test['Text']), len(card_test['negated_text']),
                                      len(epa_test['Text']), len(epa_test['negated_text'])])
        test_df['Numerical Rating'] = numerical_rating
        test_df['Synthetic'] = synthetic_test_df_array

        # Remove duplicates in train_df and test_df
        train_df = train_df.drop_duplicates(subset=['Text'])
        test_df = test_df.drop_duplicates(subset=['Text'])

        if only_use_synthetic:
            train_df = train_df[train_df['Synthetic'] == True]
            test_df = test_df[test_df['Synthetic'] == True]

        # Print value counts
        if self.verbose:
            print(train_df['Numerical Rating'].value_counts())
            print(test_df['Numerical Rating'].value_counts())

        return train_df, test_df

    def create_cv_folds(self, train_df, test_df, n_splits=2, shuffle=True, random_state=23):
        if n_splits == 1:
            return [(train_df, test_df)]

        # Concat train and test dataframes
        df = pd.concat([train_df, test_df], ignore_index=True)
        df = df.drop_duplicates(subset=['Text'])


        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        folds = []
        
        for train_index, test_index in kf.split(df):
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]
            
            if any(train_df['Text'].isin(test_df['Text'])):
                raise ValueError("A claim in 'Text' column in train_df is also in test_df.")

            folds.append((train_df, test_df))
        
        return folds

