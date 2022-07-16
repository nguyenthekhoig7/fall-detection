from random import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import learning_curve
import random


class Visualise:
    def __init__(self) -> None:
        pass

    def conf_mat(self, y_test: np.array, y_pred: np.array) -> None:
        cm = confusion_matrix(y_test, y_pred)
        norm_conf = []
        for i in cm:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j)/float(a))
            norm_conf.append(tmp_arr)

        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(1, 2, 1)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                        interpolation='nearest')

        width, height = cm.shape

        for x in range(width):
            for y in range(height):
                ax.annotate(str(cm[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

        plt.title("Confusion Matrix")
        plt.xticks(range(width), ['positive', 'negative'])
        plt.yticks(range(height), ['positive', 'negative'])
        plt.show()

    def show_roc(self, y_test: np.array, y_pred: np.array) -> None:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        plt.title('ROC curve')
        plt.plot(false_positive_rate, true_positive_rate,
                 'b', label='AUC = %0.2f' % roc_auc, marker='.')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def report(self, y_test, y_pred) -> None:
        print(classification_report(y_test, y_pred))

    def learning_curve_plot(self, data: np.array, label: np.array, model) -> None:
        train_sizes, train_scores, valid_scores = learning_curve(
            model, data, label, train_sizes=np.arange(start=1, stop=label.shape[0] - 1100, step=random.randint(500, 750), dtype=int), cv=10, scoring='accuracy', n_jobs=-1)
        train_scores = np.nan_to_num(train_scores)
        valid_scores = np.nan_to_num(valid_scores)
        train_mean = np.mean(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)
        plt.subplots(1, figsize=(10, 10))
        plt.plot(train_sizes, train_mean, label="Train score")
        plt.plot(train_sizes, valid_mean, label="Val score")
        plt.ylim(0, 1.2)
        plt.title("Learning Curve")
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy Score")
        plt.legend(loc="best")
