from data import Data
from dataPreprocessing import DataPreprocessing
from model import Model
from metrics import Metrics
from visualization import Visualization


class Main:

    def __init__(self):

        # Create objects
        self.data = Data()

        self.preprocessing = DataPreprocessing()

        self.model = Model()

        self.metrics = Metrics()

        self.visualization = Visualization()

    def run(self):

        # =================================================
        # 1. LOAD DATA
        # =================================================

        x, y = self.data.load_data()

        self.data.display_data()

        self.data.check_missing_values()


        # =================================================
        # 2. VISUALIZE DATA DISTRIBUTION
        # =================================================

        self.visualization.plot_class_distribution(y)


        # =================================================
        # 3. TRAIN TEST SPLIT
        # =================================================

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = self.preprocessing.split_data(
            x,
            y
        )


        # =================================================
        # 4. DATA PREPROCESSING
        # =================================================

        (
            X_train,
            X_test
        ) = self.preprocessing.scale_data()


        # =================================================
        # 5. TRAIN MODEL
        # =================================================

        self.model.train(
            X_train,
            y_train
        )


        # =================================================
        # 6. PREDICTION
        # =================================================

        predictions = self.model.predict(
            X_test
        )


        # =================================================
        # 7. CONFUSION MATRIX
        # =================================================

        self.metrics.calculate_confusion_matrix(
            y_test,
            predictions
        )


        # =================================================
        # 8. PERFORMANCE METRICS
        # =================================================

        self.metrics.calculate_performance(
            y_test,
            predictions
        )


        # =================================================
        # 9. DISPLAY METRICS
        # =================================================

        self.metrics.display_metrics()


        # =================================================
        # 10. CONFUSION MATRIX GRAPH
        # =================================================

        self.visualization.plot_confusion_matrix(
            self.metrics.get_confusion_matrix()
        )


        # =================================================
        # 11. ACTUAL VS PREDICTED GRAPH
        # =================================================

        self.visualization.plot_actual_predicted(
            y_test,
            predictions
        )


# =========================================================
# PROGRAM START
# =========================================================

if __name__ == "__main__":

    app = Main()

    app.run()