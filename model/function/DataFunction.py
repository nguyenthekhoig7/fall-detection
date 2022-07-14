import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class DataFunction:
    def __init__(self) -> None:
        pass

    def labelSet(self, choice: str, fallPath: str, normalPath: str) -> np.array:
        label = []
        if choice.strip().lower() == "origin":
            for file in os.listdir(fallPath):
                df = pd.read_csv(fallPath + file)
                for i in range(len(df)):
                    label.append(1)
            for file in os.listdir(normalPath):
                df = pd.read_csv(normalPath + file)
                for i in range(len(df)):
                    label.append(-1)
        if choice.strip().lower() == "addframe":
            restFallFrame = 10
            for file in os.listdir(fallPath):
                df = pd.read_csv(fallPath + file)
                for i in range(len(df)):
                    if i >= restFallFrame:
                        label.append(1)
                    else:
                        label.append(-1)
            for file in os.listdir(normalPath):
                df = pd.read_csv(normalPath + file)
                for i in range(len(df)):
                    label.append(-1)
        return np.array(label)

    def splitData(self, data: np.array, label: np.array) -> np.array:
        x_train, x_test, y_train, y_test = train_test_split(
            data, label, test_size=0.2, random_state=0, shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=.25, shuffle=True)
        return x_train, x_test, x_val, y_train, y_test, y_val

    def balanceTrainData(self, x: np.array, y: np.array) -> np.array:
        sm = SMOTE()
        x, y = sm.fit_resample(x, y)
        return x, y
