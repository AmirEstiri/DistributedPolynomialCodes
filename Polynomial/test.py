import numpy as np
from scipy import interpolate

F = 1e22

# A = [[[1, 1, 1, 3], [2, 0, 2, 1]],
#      [[0, 1, 0, 4], [0, 1, 0, 4]],
#      [[1, 4, 5, 0], [5, 4, 1, 1]],
#      [[2, 3, 2, 0], [2, 4, 1, 2]],
#      [[6, 7, 8, 0], [6, 7, 1, 9]]]
#
# B = [[[1, 1, 1, 9], [0, 1, 0, 8]],
#      [[7, 6, 8, 6], [7, 9, 5, 9]],
#      [[6, 1, 3, 5], [8, 4, 2, 9]],
#      [[7, 7, 1, 2], [6, 9, 0, 5]],
#      [[5, 8, 1, 6], [4, 9, 7, 2]]]

A = [[[1, 1], [1, 3], [2, 0], [2, 1]],
     [[0, 1], [0, 4], [0, 1], [0, 4]],
     [[1, 4], [5, 0], [5, 4], [1, 1]],
     [[2, 3], [2, 0], [2, 4], [1, 2]],
     [[6, 7], [8, 0], [6, 7], [1, 9]]]

B = [[[1, 1], [1, 9], [0, 1], [0, 8]],
     [[7, 6], [8, 6], [7, 9], [5, 9]],
     [[6, 1], [3, 5], [8, 4], [2, 9]],
     [[7, 7], [1, 2], [6, 9], [0, 5]],
     [[5, 8], [1, 6], [4, 9], [7, 2]]]

A_tildes = []
B_tildes = []
C_tildes = []
for i in range(6):
    A_tildes.append(np.zeros((5, 4)))
    B_tildes.append(np.zeros((5, 4)))
    C_tildes.append(np.zeros((4, 4)))

for k in range(1, 6):
    for i in range(5):
        for j in range(4):
            A_tildes[k - 1][i][j] = (A[i][j][0] + A[i][j][1] * (k ** 1))# + A[i][j][2] * (k ** 2) + A[i][j][3] * (k ** 3)) % F
            B_tildes[k - 1][i][j] = (B[i][j][0] + B[i][j][1] * (k ** 2))# + B[i][j][2] * (k ** 8) + B[i][j][3] * (k ** 12)) % F

for k in range(5):
    C_tildes[k] = np.matmul(A_tildes[k].T, B_tildes[k]) % F
    print "node " + str(k + 1)
    print A_tildes[k]
    print B_tildes[k]
    print C_tildes[k]

poly_y = []
poly_x = [1, 2, 3, 4]#, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
for k in range(4):
    poly_y.append(C_tildes[k][0][1])
interpolation = interpolate.lagrange(poly_x, poly_y)
print interpolation
# coeff = np.polyfit(poly_x, poly_y, deg=15)
# print coeff

# for k in range(16):
#     print "(" + str(poly_x[k]) + "," + str(poly_y[k]) + ")"
