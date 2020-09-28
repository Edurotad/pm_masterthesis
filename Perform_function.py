import numpy as np
import pandas as pd 
import sklearn 
import matplotlib.pyplot as plt
from math import exp


def perform_func(y_pred, y_test, UL_test):

    error = y_pred - y_test
    acc_v = np.zeros(len(error))
    for i in range(len(error)):
        if (error[i]>=-13)&(error[i]<=10):
            acc_v[i]=1

    accuracy = (100/len(y_test))* acc_v.sum()

    mae = np.abs(error).mean()
    mse = (1/len(y_test)*(error**2).sum())
    mape = (100/len(y_pred))*(np.abs(error)/y_pred).sum()
    mape_2 = (100/len(y_pred))*(np.abs(error)/(y_test+UL_test)).sum()

    def Computed_Score(y_true, y_pred):
    ##Computed score used in the challenge

        a1 = 10
        a2 = 13
        score = 0
        d = y_pred - y_true

        for i in d: 
            if i<0:
                score += (exp(-i/a1) - 1)
            else : 
                score += (exp(i/a2) - 1)
        
        return score

    score = Computed_Score(y_test,y_pred)

    print("The model score is: {}".format(score))
    print("The model accuracy is: {}".format(accuracy))
    print("The model MAE is: {}".format(mae))
    print("The model MSE is : {}".format(mse))
    print("The model Mean Absolute Percentage Error (MAPE) is: {}".format(mape))
    print("The model Mean Absolute Percentage Error 2 (MAPE_2) is: {}".format(mape_2))

    ## Data prediction plots for the dataset

    if (len(y_pred)<120):

        # Plot in blue color the predicted data and in green color the
        # actual data to verify visually the accuracy of the model.
        fig_verify = plt.figure(figsize=(10, 5))
        plt.plot(y_pred, color="#3f729a")
        plt.plot(y_test, color="#89ce65")
        plt.ylabel('Remaining Useful Life')
        plt.xlabel('Unit')
        plt.legend(['RUL Prediction', 'Real RUL'], loc='best')
        plt.show()

    else:

        num = round(len(y_pred)/2)

        fig_verify = plt.figure(figsize=(10, 5))
        plt.plot(range(1,num+1),y_pred[:num], color="#3f729a")
        plt.plot(range(1,num+1),y_test[:num], color="#89ce65")
        plt.ylabel('Remaining Useful Life')
        plt.xlabel('Unit')
        plt.legend(['RUL Prediction', 'Real RUL'], loc='best')
        plt.show()
        
        fig_verify = plt.figure(figsize=(10, 5))
        plt.plot(range(num+1,len(y_pred)+1),y_pred[num:], color="#3f729a")
        plt.plot(range(num+1,len(y_pred)+1),y_test[num:], color="#89ce65")
        plt.ylabel('Remaining Useful Life')
        plt.xlabel('Unit')
        plt.legend(['RUL Prediction', 'Real RUL'], loc='best')
        plt.show()

    
    # Plot every unit prediction and real value,
    # so we can see what the deviation is. 

    fig_predict = plt.figure(figsize = (10,5))
    plt.scatter(y_pred, y_test, color = "#3f729a")
    plt.plot(range(y_test.max().astype(int)),range(y_test.max().astype(int)), color = "black")
    plt.ylabel('Real value')
    plt.xlabel('Predicted value')
    plt.legend(['Accurate prediction line', 'Predicted points'])
    plt.show()

    # Plot the predictions histogram to study how
    # deviated are the predictions from the mean

    fig_hist = plt.figure(figsize=(10,5))
    plt.hist(error, color = "#3f729a" )
    plt.legend(['Error'])

    return score, accuracy, mae, mse, mape, mape_2

