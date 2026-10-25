import cv2
import numpy as np

def triangulatePoints( P1, P2, x1, x2 ):
    X = cv2.triangulatePoints( P1[:3], P2[:3], x1[:2], x2[:2] )
    return X/X[3] # Remember to divide out the 4th row. Make it homogeneous

def project(P, X): 
    x = np.dot(P[:3],X)
    return x/x[2] # Remember to divide out the 3rd row. Make it homogeneous   

# Camera projection matrices
P1 = np.eye(4)
P2 = np.array([[ 0.878, -0.01 ,  0.479, -1.995],
               [ 0.01 ,  1.   ,  0.002, -0.226],
               [-0.479,  0.002,  0.878,  0.615],
               [ 0.   ,  0.   ,  0.   ,  1.   ]])


# Homogeneous 3D points array
X4xN = np.array([[ 1.00277401,  2.00861221,  3.01259262,  1.00350119,  2.01054271],
                 [ 4.01217993,  4.01031008,  4.01743742,  4.02958919,  4.01894571],
                 [11.01977924, 12.02856882, 13.04163081, 12.09159201, 13.05497299],
                 [         1.,          1.,          1.,          1.,          1.]])


# Homogeneous image points arrays
x1_3xN = np.array([[ 0.091,  0.167,  0.231,  0.083,  0.154],
                   [ 0.364,  0.333,  0.308,  0.333,  0.308],
                   [ 1.   ,  1.   ,  1.   ,  1.   ,  1.   ]])
x2_3xN = np.array([[ 0.42 ,  0.537,  0.645,  0.431,  0.538],
                   [ 0.389,  0.375,  0.362,  0.357,  0.345],
                   [ 1.   ,  1.   ,  1.   ,  1.   ,  1.   ]])

# project P*X and override origin arrays
if True:
    x1_3xN = project(P1,X4xN)
    x2_3xN = project(P2,X4xN)

# The cv2 method
X_est = triangulatePoints( P1[:3], P2[:3], x1_3xN[:2], x2_3xN[:2] )

# Recover the origin arrays from P*X
x1 = project(P1,X_est)
x2 = project(P2,X_est)

# Again, put in homogeneous form before using them
x1 /= x1[2]
x2 /= x2[2]
 
print ('X_est\n', X_est)
print ('x1\n', x1)
print ('x2\n', x2)

err_3d = np.mean(np.absolute(X4xN - X_est)[:3])
print('err_3d: ', err_3d)  

err_2d_1 = np.mean(np.absolute(x1_3xN - x1)[:2])
print('err_2d_1: ', err_2d_1)  

err_2d_2 = np.mean(np.absolute(x2_3xN - x2)[:2])
print('err_2d_2: ', err_2d_2)  

