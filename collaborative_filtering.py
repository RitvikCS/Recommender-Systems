'''
Collaborative Filtering by :

S.Devendra Dheeraj Gupta  -  2017B5A70670H
K.Srinivas  -  2017B3A70746H
Abhirath Singh -  2018A7PS0521H
Ritvik C -  2018A7PS0180H
'''

import pandas as pd
import numpy as np
import pickle
import operator
from copy import deepcopy
from collections import Counter
from time import time

'''
    Pickle library is used to load train and test matrices, users_map and movies_map saved from preprocess.py
'''
print("Hello! Starting item - item collaborative filtering")
filehandler = open("matrix_dump", 'rb+')
matrix = pickle.load(filehandler)
n_users = matrix.shape[0]
n_movies = matrix.shape[1]

filehandler = open("test_dump", 'rb+')
test = pickle.load(filehandler)

filehandler = open("user_dump", 'rb+')
users_map = pickle.load(filehandler)

filehandler = open("movie_dump", 'rb+')
movies_map = pickle.load(filehandler)
print("Pickle files are loaded")

matrix = matrix.T    #The rows are now movies and columns are users
'''
The pearson co-efficient matrix between movies to find similarities between two movies is calculated.
'''
sim_matrix = np.corrcoef(matrix)
'''
We use the test cases to estimate the ratings using collaborative filtering (item-item)
'''
actual_ratings = []
pred_ratings = []

start_time = time()
print("Starting predictions at time stamp:", start_time)

for i in range(len(test["movieid"])):
    user = test.iloc[i,0]
    movie = test.iloc[i,1]
    rating = test.iloc[i,2]

    movie = movies_map[str(movie)]
    user = users_map[str(user)]
    actual_ratings.append(int(rating))

    num = 0
    den = 0
    sim_movie = sim_matrix[movie]
    user_ratings = matrix[:,user]
    for j in range(n_movies):
        if user_ratings[j] != 0:
            num = num + sim_movie[j]*user_ratings[j]
            den = den + sim_movie[j]
    pred_ratings.append(num/den)

end_time = time()
print("Done with predictions at times stamp:", end_time)

def RMSE(pred,value):
    '''
    RMSE functoin takes the predicted ratings and the actual ratings as parameters
    and returns the RMSE of the predictions as the result.
    '''
    N = len(pred)
    sum = np.sum(np.square(pred-value))
    return np.sqrt(sum/N)

def spearmanCorr(pred,value):
    '''
    spearmanCorr functoin takes the predicted ratings and the actual ratings as parameters
    and returns the spearman correlation of the predictions as the result.
    '''
    N = len(pred)
    sum = np.sum(np.square(pred-value))
    den = N*(N**2 - 1)
    return 1 - (6*sum)/den

def precisionAtRankK(pred,value,length):
    '''
    precisionAtRankK functoin takes the predicted ratings, actual ratings and
    number of ratings as parameters and returns the precision of the predictions as the result.
    A threshold rating of 3 has been used to identify valid predictions.
    '''
    den = 0
    num = 0

    for i in range(length):
        if pred[i]>3 and value[i]>3:
            num = num + 1
            den = den + 1
        elif pred[i]>3:
            den = den + 1

    return num/den

print("RMSE:",RMSE(np.array(pred_ratings), np.array(actual_ratings)))
print("Precision on top K", precisionAtRankK(np.array(pred_ratings), np.array(actual_ratings), len(pred_ratings)))
print("Spearman Correlation:",spearmanCorr(np.array(pred_ratings), np.array(actual_ratings)))
print("Prediction time:",end_time - start_time)
