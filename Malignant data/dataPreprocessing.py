from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessing:

    def __init__(self, test_size=0.2, random_state=42):

        self.test_size = test_size
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.scaler = StandardScaler()

    def split_data(self, x, y):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(

            x,
            y,

            test_size=self.test_size,

            random_state=self.random_state,

            stratify=y
        )

        print("\n========== TRAIN TEST SPLIT ==========")

        print("Training data:", self.X_train.shape)
        print("Testing data :", self.X_test.shape)

        return (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        )

    def scale_data(self):

        # Fit only on training data
        self.X_train = self.scaler.fit_transform(self.X_train)

        # Transform testing data
        self.X_test = self.scaler.transform(self.X_test)

        print("\n========== DATA SCALING ==========")

        print("StandardScaler applied successfully.")

        return self.X_train, self.X_test