import pandas as pd
import numpy as np


class DataLoader():
    def __init__(self, percentage_false: float):
        np.random.seed(seed=23)
        self.percentage_false = percentage_false
        # Ground Truth
        self.ground_truth = pd.read_csv(
            "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Raw Data/Climate/VK's Copy of Cleaned - Google Fact Check Explorer - Climate.xlsx - Corrected.csv")
        self.ground_truth = self.ground_truth.dropna(subset=['Text'])
        self.ground_truth = self.ground_truth[self.ground_truth['Text'] != '']

        # EPA
        self.epa_who_data = pd.read_csv(
            "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Raw Data/Climate/climate_change_epa_who.csv")
        self.epa_who_data['Category'] = -1
        self.epa_who_data['Numerical Rating'] = 3
        self.epa_who_data = self.epa_who_data.dropna(subset=['Text'])
        self.epa_who_data = self.epa_who_data[self.epa_who_data['Text'] != '']

        self.epa_who_data_negated = pd.read_csv(
            "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Raw Data/Climate/Negated Claims/negated_epa_data.csv")
        self.epa_who_data_negated['Category'] = -1
        self.epa_who_data_negated['Numerical Rating'] = 1
        self.epa_who_data_negated = self.epa_who_data_negated.rename(columns={'text': 'Text'})
        self.epa_who_data_negated = self.epa_who_data_negated.dropna(subset=['Text'])
        self.epa_who_data_negated = self.epa_who_data_negated[self.epa_who_data_negated['Text'] != '']



        # Card Data
        self.card_data = pd.read_csv(
            "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Raw Data/Climate/card_train_with_score.csv")
        self.card_data['claim'] = '1_1'
        self.card_data = self.card_data.rename(columns={'text': 'Text', 'claim': 'Category'})
        # To the card_data, add a 'Numerical Rating' column with value 1
        self.card_data['Numerical Rating'] = 1
        self.card_data = self.card_data.dropna(subset=['Text'])
        self.card_data = self.card_data[self.card_data['Text'] != '']
        self.card_data = self.card_data[self.card_data['score'] > 0.7]

        self.card_data_negated = pd.read_csv(
            "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Raw Data/Climate/Negated Claims/negated_card_data_score_over_0.7.csv")
        self.card_data_negated['claim'] = '1_1'
        self.card_data_negated = self.card_data_negated.rename(columns={'text': 'Text', 'claim': 'Category'})
        # To the card_data, add a 'Numerical Rating' column with value 1
        self.card_data_negated['Numerical Rating'] = 3
        # Drop the Unnamed: 0 column
        self.card_data_negated = self.card_data_negated.drop(columns=['Unnamed: 0'])
        self.card_data_negated = self.card_data_negated.dropna(subset=['Text'])
        self.card_data_negated = self.card_data_negated[self.card_data_negated['Text'] != '']
        self.card_data_negated = self.card_data_negated[self.card_data_negated['score'] >= 0.7]


    def create_train_test_df(self, use_card_data: bool, use_epa_data: bool, use_ground_truth: bool) -> (
    pd.DataFrame, pd.DataFrame):

        # Filter card_data where 'Category' is not 0_0
        card_data_claim_categories = len(self.card_data['Category'].value_counts())

        num_per_category_needed = 0
        if use_epa_data:
            num_per_category_needed = int((self.percentage_false / (1 - self.percentage_false) * len(self.epa_who_data)) // card_data_claim_categories) + 1
        elif use_ground_truth:
            ground_truth_false = len(self.ground_truth[self.ground_truth['Numerical Rating'] == 1])
            ground_truth_true = len(self.ground_truth[self.ground_truth['Numerical Rating'] == 3])
            num_per_category_needed = int(((self.percentage_false / (1 - self.percentage_false) * ground_truth_true) - ground_truth_false) // card_data_claim_categories) + 1
            if num_per_category_needed < 0:
                num_per_category_needed = 0

        card_data_new = self.card_data.groupby('Category').head(num_per_category_needed)
        # If len(self.card_data) is less than num_per_category_needed * card_data_claim_categories, then add more
        # rows to self.card_data
        while num_per_category_needed * card_data_claim_categories > len(card_data_new):
            card_data_new = pd.concat([card_data_new, self.card_data.sample()])
            card_data_new = card_data_new.drop_duplicates()

        self.card_data = card_data_new
        # Filter to rows where 'Numerical Rating' is 1 or 3
        self.ground_truth = self.ground_truth[self.ground_truth['Numerical Rating'].isin([1, 3])]

        # Drop all columns except 'Text' and 'Numerical Rating'
        self.ground_truth = self.ground_truth[['Text', 'Numerical Rating']]
        # Concat epa_who_data and ground_truth
        # ground_truth = pd.concat([ground_truth, epa_who_data], ignore_index=True)

        # Print number of 1s and 3s
        # print(self.ground_truth['Numerical Rating'].value_counts())
        # print(self.card_data['Numerical Rating'].value_counts())
        # print(self.epa_who_data['Numerical Rating'].value_counts())
        objects = []
        if use_card_data:
            objects.append(self.card_data)
        if use_epa_data:
            objects.append(self.epa_who_data)
        if use_ground_truth and use_epa_data:
            count_false = len(self.ground_truth[self.ground_truth['Numerical Rating'] == 1])
            while num_per_category_needed * card_data_claim_categories > count_false:
                sample = self.card_data.sample()
                sample['Numerical_Rating'] = [1]
                self.ground_truth = pd.concat([self.ground_truth, sample])
                self.ground_truth = self.ground_truth.drop_duplicates()
                count_false = len(self.ground_truth[self.ground_truth['Numerical Rating'] == 1])
            objects.append(self.ground_truth)
        if use_ground_truth and use_card_data:
            objects.append(self.ground_truth)
        train_df = pd.concat(objects, ignore_index=True)
        # Select random sample of 80% such that there is an even distribution of 1s and 3s
        test_df = train_df.groupby('Numerical Rating').apply(lambda x: x.sample(frac=0.2)).reset_index(drop=True)

        print(train_df['Numerical Rating'].value_counts())
        print(test_df['Numerical Rating'].value_counts())
        return train_df, test_df

    def create_large_train_test_df(self) -> (pd.DataFrame, pd.DataFrame):
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
        train_df = train_df[train_df['length'] > train_df['length'].quantile(0.1)]
        train_df = train_df[train_df['length'] < train_df['length'].quantile(0.9)]
        test_df = test_df[test_df['length'] > test_df['length'].quantile(0.1)]
        test_df = test_df[test_df['length'] < test_df['length'].quantile(0.9)]

        print(train_df['Numerical Rating'].value_counts())
        print(test_df['Numerical Rating'].value_counts())

        return train_df, test_df


