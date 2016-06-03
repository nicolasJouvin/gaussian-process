# -*-encoding: utf-8 -


import numpy as np
import pandas as pd
from sklearn import gaussian_process
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df = pd.read_excel('C:/users/Nicolas/PycharmProjects/Concrete_Data.xls', header=0).dropna()
    theta_0=1
    theta_L=1e-4    # setting hyper-parameters
    theta_U=1e-3
    gp = gaussian_process.GaussianProcess(theta0=theta_0, thetaL=theta_L, thetaU=theta_U)

    X = np.array(df.iloc[:30,[0]]) + np.random.normal(size=(30,1), loc=0, scale=5) # adding some noise because Gp don't allow                                                                                # equality in input.
    y = np.array(df.iloc[:30,[8]])                                                 # last column of the file is the value of interest

    """print(X.shape)
    print(X)
    print(y.shape)
    plt.hist(X)
    plt.hist(y.as_matrix())
    plt.show()"""

    """ Some trial to remove equality in X (got to try method .groupby of panda)

    Xu = np.unique(X)
    print(Xu)
    yu=np.extract([X[i]==x1 for i,x1 in enumerate(Xu)] , np.array(df.iloc[:30,[8]]))
    print(Xu.shape)
    print(yu)
    print(X,y)
    unique_index = [np.where(X==x1) for x1 in Xu]    # get corresponding indices for unique X values
    print(Xu, y[unique_index])"""

    gp.fit(X, y)    # fitting the feature 1 on Y with the GP
    n = 1000    # number of points to predict (the more the smoother)
    x = np.atleast_1d(np.linspace(X.min(), X.max(), n)).reshape((n, 1))  # n input values for prediction
    y_pred = gp.predict(x, eval_MSE=False)  # prediction on x with the GP
    y_pred = np.array(y_pred).T
    print(y_pred.shape)
    print("y_pred=",y_pred)
    print(y_pred[0])
    temp= np.linspace(X.min(), X.max(), n)
    temp2= y_pred[0]
    print(temp.shape, temp2.shape)
    plt.plot(np.linspace(X.min(), X.max(), n), y_pred[0], 'b')
    plt.scatter(X, y)
    plt.title('theta_0=%s     '%theta_0 + 'theta_L=%s     '%theta_L + 'theta_U=%s    '%theta_U)
    plt.savefig('Image Gp/Concrete_feature1_GP_theta0=%s'%theta_0 + 'thetaL=%s'%theta_L + 'thetaU=%s'%theta_U + '.png')
    plt.show()

