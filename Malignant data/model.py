from sklearn.linear_model import LogisticRegression


class Model:

    def __init__(self):

        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        self.predictions = None

    def train(self, X_train, y_train):

        self.classifier.fit(
            X_train,
            y_train
        )

        print("\n========== MODEL TRAINING ==========")

        print("Logistic Regression model trained successfully.")

    def predict(self, X_test):

        self.predictions = self.classifier.predict(
            X_test
        )

        print("\n========== PREDICTION ==========")

        print("Prediction completed successfully.")

        return self.predictions

    def get_model(self):

        return self.classifier