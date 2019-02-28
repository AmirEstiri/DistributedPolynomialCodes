import csv
import numpy as np


def read_data_cancer():
    with open('Data/cancer/train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        X = []
        y = []
        for row in csv_reader:
            int_row = np.array(row).astype(np.int)
            X.append(int_row[1:-1])
            y.append(int((2-int_row[-1])/2))
    X = np.array(X)
    y = np.array(y)
    return X, y


X, y = read_data_cancer()
print(X[0])