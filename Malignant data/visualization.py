import matplotlib.pyplot as plt
import numpy as np


class Visualization:

    def __init__(self):
        pass

    def plot_class_distribution(self, y):

        malignant = (y == 1).sum()
        benign = (y == 0).sum()

        labels = [
            "Benign",
            "Malignant"
        ]

        values = [
            benign,
            malignant
        ]

        plt.figure(figsize=(7, 5))

        plt.bar(
            labels,
            values
        )

        plt.title("Breast Cancer Class Distribution")

        plt.xlabel("Class")

        plt.ylabel("Number of Samples")

        plt.show()

    def plot_confusion_matrix(self, cm):

        plt.figure(figsize=(6, 5))

        plt.imshow(cm)

        plt.title("Confusion Matrix")

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.xticks(
            [0, 1],
            ["Benign", "Malignant"]
        )

        plt.yticks(
            [0, 1],
            ["Benign", "Malignant"]
        )

        # Display values inside matrix
        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        plt.colorbar()

        plt.show()

    def plot_actual_predicted(self, y_test, predictions):

        actual_counts = [
            np.sum(y_test == 0),
            np.sum(y_test == 1)
        ]

        predicted_counts = [
            np.sum(predictions == 0),
            np.sum(predictions == 1)
        ]

        labels = [
            "Benign",
            "Malignant"
        ]

        x = np.arange(len(labels))

        width = 0.35

        plt.figure(figsize=(7, 5))

        plt.bar(
            x - width / 2,
            actual_counts,
            width,
            label="Actual"
        )

        plt.bar(
            x + width / 2,
            predicted_counts,
            width,
            label="Predicted"
        )

        plt.xticks(
            x,
            labels
        )

        plt.xlabel("Class")

        plt.ylabel("Number of Samples")

        plt.title("Actual vs Predicted Classes")

        plt.legend()

        plt.show()