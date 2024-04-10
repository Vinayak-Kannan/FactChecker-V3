import pandas as pd


class DataLoader():
    def __init__(self, percentage_false: float):
        self.percentage_false = percentage_false
        self.ground_truth = pd.read_csv(
            "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Raw Data/VK's Copy of Cleaned - Google Fact Check Explorer - Climate.xlsx - Corrected.csv")
        self.epa_who_data = pd.read_csv(
            "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Scraping/Transformed Data/climate_change_epa_who.csv")
        self.epa_who_data['Category'] = -1

        self.card_data = pd.read_csv(
            "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Testing/Adhoc Analysis/data/training/training.csv")

        self.card_data_score = pd.read_csv(
            "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Testing/Adhoc Analysis/data/training/train_with_score.csv")
        self.card_data['score'] = self.card_data_score['score']

    def create_train_test_df(self, use_card_data: bool, use_epa_data: bool, use_ground_truth: bool) -> (
    pd.DataFrame, pd.DataFrame):
        # To the card_data, add a 'Numerical Rating' column with value 1
        self.card_data['Numerical Rating'] = 1  # 10 * 3/2 // 5
        # Filter card_data where 'Category' is not 0_0
        self.card_data = self.card_data.rename(columns={'text': 'Text', 'claim': 'Category'})
        self.card_data = self.card_data[self.card_data['Category'] != '0_0']
        self.card_data = self.card_data[self.card_data['score'] > 0.7]
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
                sample = self.card_data.sample().rename(columns={'text': 'Text', 'claim': 'Category'})
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
