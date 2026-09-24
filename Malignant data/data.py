import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer


class Data:

    def __init__(self):
        self.data = None
        self.x = None
        self.y = None

    def load_data(self):

        self.data = load_breast_cancer()

        
        self.x = pd.DataFrame(
            self.data.data,
            columns=self.data.feature_names
        )


        self.y = pd.Series(
            (self.data.target == 0).astype(int),
            name="malignant"
        )

        return self.x, self.y

    def display_data(self):

        print("\n========== DATASET ==========")

        print("Number of rows:", self.x.shape[0])
        print("Number of columns:", self.x.shape[1])

        print("\nFirst 5 records:")
        print(self.x.head())

        print("\nTarget values:")
        print(self.y.head())

        print("\nTarget distribution:")
        print(self.y.value_counts())

    def check_missing_values(self):

        missing = self.x.isnull().sum().sum()

        print("\n========== MISSING VALUES ==========")

        print("Total missing values:", missing)

        if missing == 0:
            print("No missing values found.")
        else:
            print("Missing values found.")

