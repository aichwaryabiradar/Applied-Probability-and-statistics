import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class Metrics:

    def __init__(self):

        self.cm = None

        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None

    def calculate_confusion_matrix(self, y_test, predictions):

        self.cm = confusion_matrix(
            y_test,
            predictions
        )

        return self.cm

    def calculate_performance(self, y_test, predictions):

        self.accuracy = accuracy_score(
            y_test,
            predictions
        )

        self.precision = precision_score(
            y_test,
            predictions
        )

        self.recall = recall_score(
            y_test,
            predictions
        )

        self.f1 = f1_score(
            y_test,
            predictions
        )

        return (
            self.accuracy,
            self.precision,
            self.recall,
            self.f1
        )

    def display_metrics(self):

        print("\n========== CONFUSION MATRIX ==========")

        print(self.cm)

        print("\nConfusion Matrix format:")

        print("                 Predicted")
        print("                 0       1")
        print("Actual    0      TN      FP")
        print("          1      FN      TP")

        print("\n========== PERFORMANCE METRICS ==========")

        print(
            "Sklearn Accuracy : ",
            self.accuracy
        )

        print(
            "Sklearn Precision: ",
            self.precision
        )

        print(
            "Sklearn Recall   : ",
            self.recall
        )

        print(
            "Sklearn F1 Score : ",
            self.f1
        )

    def get_confusion_matrix(self):

        return self.cm

    def get_performance(self):

        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1
        }