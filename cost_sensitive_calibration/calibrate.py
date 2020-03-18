import numpy as np
from sklearn.metrics import roc_curve


class BinaryCalibration:
    """Calibrate binary classifiers."""

    def __init__(self, utility_matrix):
        """
        Initialize the calibration.

        :param utility_matrix: dictionary defining tp, fp, fn, tn unitary utility
        """
        self.util_matrix = utility_matrix

    def utility(self, fpr, tpr, positives_freq):
        """Compute the utility given a specific value of fpr, tpr and positives frequency."""
        a = self.util_matrix["tp"]
        b = self.util_matrix["fp"]
        c = self.util_matrix["fn"]
        d = self.util_matrix["tn"]
        negatives_freq = 1.0 - positives_freq
        return (
            positives_freq * (a - c) * tpr
            + negatives_freq * (b - d) * fpr
            + c * positives_freq
            + d * negatives_freq
        )

    def calibration_curve_tpr(self, fpr, positives_freq, util_delta=0.0):
        """Compute the the true positive rate along the calibration curve in terms of utility."""
        a = self.util_matrix["tp"]
        b = self.util_matrix["fp"]
        c = self.util_matrix["fn"]
        d = self.util_matrix["tn"]
        negatives_freq = 1.0 - positives_freq
        return (
            negatives_freq * (b - d) * fpr
            + c * positives_freq
            + d * negatives_freq
            - util_delta
        ) / ((c - a) * positives_freq)

    def __estimate_optimal_threshold_and_utility(self, fpr, tpr, th, positives_freq):
        """We crunch the numbers here."""
        delta = tpr - self.calibration_curve_tpr(fpr, positives_freq)
        max_delta_index = np.where(delta == np.amax(delta))[0][0]
        max_util = self.utility(
            fpr[max_delta_index], tpr[max_delta_index], positives_freq
        )
        return th[max_delta_index], max_util

    def calibrate(self, labels, pred, plot_roc=False):
        """
        Calibrate the binary classifier.

        :param labels: array of binary values
        :param pred: array of float values
        :param plot_roc: boolean
        :return: return optimal threshold, max utility
        """
        fpr, tpr, th = roc_curve(labels, pred)
        positives_freq = np.sum(labels) / float(len(labels))
        optimal_th, max_utility = self.__estimate_optimal_threshold_and_utility(
            fpr, tpr, th, positives_freq
        )

        if plot_roc:
            self.__plot_roc(fpr, tpr, max_utility, optimal_th, positives_freq)

        return optimal_th, max_utility

    def __plot_roc(self, fpr, tpr, max_utility, optimal_th, positives_freq):
        import matplotlib.pyplot as plt

        steps = np.linspace(0.0, 1.0, 100)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            label="utility = %0.6f, opt th = %0.4f" % (max_utility, optimal_th),
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.plot(
            steps,
            self.calibration_curve_tpr(steps, positives_freq),
            color="red",
            lw=1,
            linestyle="--",
        )
        plt.plot(
            steps,
            self.calibration_curve_tpr(steps, positives_freq, max_utility),
            color="green",
            lw=1,
            linestyle="--",
        )
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()
        return


class BinaryBayesianMinimumRisk:
    """Bayesiann minimum risk is in charge to select best action given many options."""

    def __init__(self, utility_matrix):
        """
        Initialize the calibration.

        :param utility_matrix: numpy matrix representing unitary utility
        """
        self.util_matrix = utility_matrix

    def predict(self, pred):
        """
        Predict pick the optimal action.

        :param pred: float representing probability of positives
        :return: integer index of the option to pick
        """
        return np.argmax(
            np.sum(self.util_matrix * np.array([[pred, 1 - pred]]), axis=1)
        )


class AcceptReviewReject:
    """Implement the class for basic accept-review-reject pipeline. Essentially
    performs a grid search for the best two thresholds in order to maximize the utility."""

    def __init__(self, utility_matrix, steps=1000):
        """
        Initialize the calibration.

        :param utility_matrix: numpy dictionary representing unitary utility with the keys
            PR, NR, PM, NM, PA, NA where:

            PR -> Positive Rejected
            NR -> Negative Rejected
            PM -> Positive to Manual Review
            NM -> Negative to Manual Review
            PA -> Positive Accepted
            NA -> Negative Accepted

        :param steps: Int representing how many steps we want perform in the grid search for each parameter.
        """
        self.util_matrix = np.array([[utility_matrix['PR'], utility_matrix['NR']],
                                     [utility_matrix['PM'], utility_matrix['NM']],
                                     [utility_matrix['PA'], utility_matrix['NA']]])
        self.steps = round(steps - 1)
        self.grid_search_results = None

    @staticmethod
    def __prob_to_pred(low_step, high_step, pred):
        high_msk = (pred > high_step) * 1
        low_msk = (low_step <= pred) * 1
        return -1 + high_msk + low_msk

    def __utility(self, low_step, high_step, labels, pred):
        N = labels.shape[0]
        prediction_label = self.__prob_to_pred(low_step, high_step, pred)
        FA = np.sum((prediction_label == -1) * labels * 1)
        FM = np.sum((prediction_label == 0) * labels * 1)
        FR = np.sum((prediction_label == 1) * labels * 1)
        NFA = np.sum((prediction_label == -1) * np.logical_not(labels) * 1)
        NFM = np.sum((prediction_label == 0) * np.logical_not(labels) * 1)
        NFR = np.sum((prediction_label == 1) * np.logical_not(labels) * 1)
        U = np.array([[FR, NFR], [FM, NFM], [FA, NFA]]) * self.util_matrix
        u = np.sum(U) / N
        return [u, low_step, high_step]

    def calibrate(self, labels, preds):
        """Calibrate the classifier thresholds.

        :returns lower threshold, higher threshold, max utility"""
        delta = np.insert(np.cumsum([1 / self.steps] * self.steps), 0, 0)
        low_steps = np.repeat(delta, delta.shape[0])
        high_steps = np.tile(delta, delta.shape[0])
        high_steps = high_steps * (high_steps >= low_steps) + low_steps * (high_steps < low_steps)
        thresholds = np.unique(np.array([low_steps, high_steps]).T, axis=0)
        low_steps, high_steps = thresholds.T
        max_util_matrix = [self.__utility(low_step, high_step, labels, preds) for low_step, high_step in zip(low_steps, high_steps)]
        self.grid_search_results = np.array(max_util_matrix)
        max_util_index = np.argmax(self.grid_search_results, axis=0)[0]
        return max_util_matrix[max_util_index][1], \
               max_util_matrix[max_util_index][2], \
               max_util_matrix[max_util_index][0]
