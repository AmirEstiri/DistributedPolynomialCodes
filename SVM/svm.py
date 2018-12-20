import csv
import numpy as np
from cvxopt import solvers, matrix


def kernel(x1, x2):
    return np.dot(x1, x2)


def transform(x):
    return x


def read_train_data():
    x = []
    y = []
    C = 10
    mean_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, ]
    row_num = -1
    with open('datasets/cancer/train.csv', 'r') as train_csv:
        readCSV = csv.reader(train_csv, delimiter=',')
        for row in readCSV:
            length = len(row)
            row_num = row_num + 1
            x_i = []
            for j in range(1, length - 1):
                x_ij = row[j]
                if x_ij != '?':
                    x_ij = int(x_ij)
                    mean_x[j - 1] = (mean_x[j - 1] * row_num + x_ij) / (row_num + 1)
                else:
                    x_ij = int(mean_x[j - 1])
                x_i.append(x_ij)
            y_i = row[length - 1]
            x.append(x_i)
            y.append(3 - int(y_i))
    svm(x, y, C)


def svm(x, y, C):
    P = []
    data_size = len(x)
    for i in range(0, data_size):
        P_i = []
        for j in range(0, data_size):
            P_ij = y[i] * y[j] * kernel(x[i], x[j])
            P_i.append(P_ij)
        P.append(P_i)
    P = matrix(np.array(P), tc='d')

    A = matrix(np.array(y).reshape(1, -1), tc='d')
    b = matrix(np.zeros(1))
    q = matrix(np.ones(data_size) * -1, tc='d')
    G = matrix(np.vstack((np.diag(np.ones(data_size) * -1), np.identity(data_size))))
    h = matrix(np.hstack((np.zeros(data_size), np.ones(data_size) * C)))

    sol = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
    alpha = np.array(sol['x'])

    w = alpha[0] * y[0] * np.array(x[0])
    for i in range(1, data_size):
        w = w + alpha[i] * y[i] * np.array(x[i])

    w0 = 0
    for s in range(0, data_size):
        if alpha[s] > 0.01:
            w0 = y[s] - np.dot(np.array(w), x[s])
            break
    test_data(w, w0)


def test_data(w, w0):
    x = []
    y = []
    mean_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, ]
    row_num = -1
    with open('datasets/cancer/test.csv', 'r') as test_csv:
        readCSV = csv.reader(test_csv, delimiter=',')
        for row in readCSV:
            length = len(row)
            row_num = row_num + 1
            x_i = []
            for j in range(1, length - 1):
                x_ij = row[j]
                if x_ij != '?':
                    x_ij = int(x_ij)
                    mean_x[j - 1] = (mean_x[j - 1] * row_num + x_ij) / (row_num + 1)
                else:
                    x_ij = int(mean_x[j - 1])
                x_i.append(x_ij)
            y_i = row[length - 1]
            x.append(x_i)
            y.append(3 - int(y_i))
    data_size = len(x)
    y_guess = []
    for k in range(0, data_size):
        print(w0 + np.dot(w, transform(x[k])))
        y_guess.append(np.sign(w0 + np.dot(w, transform(x[k]))))
    print(0.5 + np.dot(y_guess, y) / (2 * data_size))


if __name__ == '__main__':
    read_train_data()
