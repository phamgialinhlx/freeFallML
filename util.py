import copy
from unittest import result
import numpy as np

def column(X, i):
    return [row[i] for row in X]

def computeCost(t, y, c):
    predict = np.dot(t,c)
    predict = np.array(predict, dtype=np.float128)
    predict = np.subtract(predict, y)
    predict = [np.power(p, 2) for p in predict]
    cost = sum(np.multiply(1 / 2 / len(t), predict))
    return cost

def gradientDescent(X, y, theta, alpha, epoch, batch_size):
    J = []
    degree = len(X[0])
    for i in range(epoch):
        miniBatches = createBatch(X, y, batch_size)
        for j in range(len(miniBatches)):
            X_mini = np.transpose(np.transpose(miniBatches[j])[0:degree])
            y_mini = column(miniBatches[j], degree)
            predict = np.dot(X_mini, theta)
            cost = np.subtract(predict, y_mini)
            tmp = np.dot(np.transpose(X_mini), cost)
            tmp = np.multiply(alpha / len(X_mini), tmp)
            theta = np.subtract(theta, tmp)
        J.append(computeCost(X, y, theta))
    return theta, J

def r_squared(y, predict):
    tmp1 = np.subtract(y, predict)
    tmp1 = [np.power(tmp, 2) for tmp in tmp1]
    tmp2 = np.subtract(y, sum(y) / len(y))
    tmp2 = [np.power(tmp, 2) for tmp in tmp2]
    return 1 - sum(tmp1) / sum(tmp2)

def poly(X, degree:int):
    Xpoly = []
    for i in range(degree + 1):
        Xpoly.append([np.power(x, i) for x in X])
    Xpoly = np.transpose(Xpoly)
    return Xpoly

# def featureScalingMatrix(X):
#     tmpX = np.transpose(X)
#     result = []
#     for i in range(len(tmpX)):
#         row = tmpX[i]
#         if (max(row) - min(row) != 0):
#             row = [((x - sum(row) / len(row)) / (max(row) - min(row))) for x in row]
#         result.append(row)
#     return np.transpose(result)

def featureScaling(t):
    result = copy.deepcopy(t)
    diff = max(t) - min(t)
    mean = sum(result) / len(result)
    # if (diff != 0):
    result = [((x - mean) / diff) for x in result]
    return result, mean, diff

def createBatch(X, y, batch_size):
    miniBatches = []
    data = np.column_stack((X, y))
    np.random.shuffle(data)
    lastBatchNumber = len(X) % batch_size
    length = int((len(X) - lastBatchNumber) / batch_size)
    for i in range(length):
        miniBatches.append(data[i * batch_size : (i + 1) * batch_size])
    if (lastBatchNumber != 0):    
        miniBatches.append(data[length * batch_size : ])
    return miniBatches