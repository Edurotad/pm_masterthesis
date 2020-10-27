import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import metrics
import math
import umap



def UMAP_model (train_v, test_v, RUL_train, RUL_test, NN = 5, UMAP_NN = 15, UMAP_dist = 0.1, n_components = 2, images = True, metric = 'euclidean'):

    import umap

    print(' ')
    print(' ')
    print('UMAP Parameters tuning: ')
    print('UMAP near neighbours = ', UMAP_NN, ', UMAP minimum distance = ', UMAP_dist)
    print('UMAP components = ', n_components, ', UMAP metric = ', metric)

    trans = umap.UMAP(
        n_neighbors = UMAP_NN,
        min_dist = UMAP_dist,
        n_components = n_components,
        metric = metric).fit(train_v)

    print('I am done')

    train_emb = trans.embedding_
    test_emb = trans.transform(test_v)

    distances = metrics.pairwise_distances(test_emb,train_emb)

    RUL_predict = np.zeros(len(test_emb))

    for i in range(len(test_emb)):

        #print('Unidad:', i)

        maxs = np.zeros(NN)

        dist = np.zeros((len(train_emb),2))
        dist[:,0] = range(len(train_emb))
        dist[:,1] = distances[i,:]

        #display(dist)
        #print(dist.shape)

        #display(np.sort(dist[:,1]))

        for j in range(NN):

            #print(np.argmin(dist[:,1]))
            #print(dist[np.argmin(dist[:,1]),1])
            

            maxs[j] = dist[np.argmin(dist[:,1]),0]

            dist = np.delete(dist, obj = np.argmin(dist[:,1]), axis = 0)
        
        RUL_predict[i] = round(np.mean(RUL_train[maxs.astype(int)]))
        #print("Maxs: ", maxs)
        #print(RUL_train[maxs.astype(int)])

    print('RUL prediction done')
    
    # Mean Absolute Error
    error = RUL_predict - RUL_test.to_numpy()[:,0]
    abs_error = np.absolute(error)
    mean_error = np.mean(abs_error)

    #Challenge score
    s = np.zeros(len(RUL_predict))

    a_1 = 10
    a_2 = 13

    for n in range(len(error)):

        if error[n] < 0 : 

            s[n] = math.exp(-error[n]/a_1) - 1 
        else : 
            s[n] = math.exp(error[n]/a_2) - 1

    performance_evaluation = np.sum(s)

    if images == True : 

        max_plot = round(len(train_emb)*0.2).astype(int)

        plt.style.use('dark_background')

        plt.scatter(train_emb[:max_plot,0], train_emb[:max_plot,1], c = RUL_train[:max_plot], cmap= 'autumn', marker = '.', label = 'Remaining Useful Life')

        plt.show()

        plt.figure(figsize = (50,10))
        plt.plot(range(len(test_emb)), RUL_predict, color = 'white', label = 'Prediction')
        plt.plot(range(len(test_emb)), RUL_test, color = 'green', label = 'Real')
        plt.plot(range(len(test_emb)), abs_error, color = 'red', label = 'Error')

        plt.legend(fontsize = 'large')

        plt.show()

        plt.scatter(RUL_predict, RUL_test, c = 'white', marker = '.')
        plt.plot(range(len(RUL_predict)), range(len(RUL_predict)), c = 'gold')

        plt.ylabel('Real RUL', size = 15)
        plt.xlabel('Predicted RUL', size = 15)

        plt.show()

    print('The Mean Absolute Error is ', mean_error)
    print('The challenge score is', performance_evaluation)

    return train_emb, test_emb, RUL_predict, mean_error, performance_evaluation
