from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from polynomialRegression import PolynomialRegression
from util import createBatch, r_squared

if __name__ == "__main__":
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    epoch = 100
    alpha = 0.01

    Xtrain = list(train.t)
    Ytrain = list(train.y)
    Xtest = list(test.t)
    Ytest = list(test.y)

    # plt.plot(Xtrain, Ytrain, ".")
    # plt.show()

    degree = []
    theta = []
    rmse = []

    # # print(createBatch([1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], 1))
    for i in range(11):
        pr = PolynomialRegression(Xtrain, Ytrain, i, alpha, epoch, 1)
        predict = pr.predict(Xtest)
        degree.append(i)
        theta.append(pr.theta)
        rmse.append(r_squared(Ytest, predict))

    df = pd.DataFrame({"Degree": degree, "Theta": theta, "RMSE": rmse})
    df.to_csv("./result.csv", index=False)

    # pr = PolynomialRegression(Xtrain, Ytrain, 2, alpha, epoch, 1)
    # predict = pr.predict(Xtest)
    
    
    # print(2,". THETA = ", pr.theta, " ", r_squared(Ytest, predict))
    # plt.figure() 
    # plt.plot(Xtest, Ytest, "x")
    # plt.plot(Xtest, predict, ".")
    # plt.show()
    # plt.figure()
    # plt.plot(pr.J)
    # plt.show()

