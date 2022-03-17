import copy
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from util import featureScaling, gradientDescent, poly
class PolynomialRegression:
    def __init__(self, X, y, degree, alpha, epoch, batch_size):
        self.X = copy.deepcopy(X)
        self.X = np.array(self.X, np.float128)
        self.X, self.meanX, self.diffX = featureScaling(self.X)   
        self.y = copy.deepcopy(y)
        self.y = np.array(self.y, np.float128)
        self.y, self.meanY, self.diffY = featureScaling(self.y)
        self.X = poly(copy.deepcopy(self.X), degree)
        self.degree = degree
        self.alpha = alpha
        self.epoch = epoch
        self.batch_size = batch_size
        self.theta, self.J = self.train()

        # X_norm = [(x - self.meanX) / self.diffX for x in X]
        # plt.plot(X_norm, np.dot(self.X, self.theta), ".")
        # plt.show()

        # plt.figure()
        # predict = self.predict(X)
        # plt.plot(X, y, "x")
        # t = np.dot(self.X, self.theta)
        # yy = np.dot(poly(X_norm, self.degree), self.theta)
        # # plt.plot(X, np.dot(poly(X_norm, self.degree), self.theta), ".")
        # plt.plot(X, [(y *self.diffY + self.meanY) for y in t], ".")
        # plt.show()


    def train(self):
        return gradientDescent(self.X, self.y, np.full(self.degree + 1, 0), self.alpha, self.epoch, self.batch_size)

    def predict(self, X):
        X_norm = [(x - self.meanX) / self.diffX for x in X]
        Y_norm = np.dot(poly(X_norm, self.degree), self.theta)
        return [(y * self.diffY + self.meanY) for y in Y_norm]
