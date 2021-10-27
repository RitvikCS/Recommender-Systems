import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

"""
Loading the data from a .dat file into a Pandas dataframe
"""
filename = 'ratings.dat'
list = []
N = 1000209
columns = ['userid','movieid','rating','timestamp']
with open(filename,'r') as f:
    for i in range(N):
        line = f.readline().strip()
        l = line.split("::")
        list.append(l)
f.closed
df = pd.DataFrame(list,columns=columns)
df.drop('timestamp',axis=1,inplace=True)
train, test = train_test_split(df,test_size=0.1)	#spliting the data into train and test. Testing dats is 10% of total data

"""
Not all movies have been rated and to eliminate discrepancies, the movies and users are mapped
"""
movies_list = df['movieid'].unique()
n_movies = len(movies_list)
movies_map = {}
for i, j in enumerate(movies_list):
    movies_map[j] = i
users_list = df['userid'].unique()
n_users = len(users_list)
users_map = {}
for i, j in enumerate(users_list):
    users_map[j] = i

"""
The 2d array, matrix, is the sparse representation of the data, with users as rows and movies as columns
"""
matrix = np.zeros([n_users, n_movies])
for index, row in df.iterrows():
    matrix[users_map[row['userid']], movies_map[row['movieid']]] = int(row['rating'])

"""
	Using pickle library, the matrices, dataframes and maps are stored to be used by another python file
"""
filehandler = open("matrix_dump",'wb+')
pickle.dump(matrix,filehandler)

filehandler = open("train_dump",'wb+')
pickle.dump(train,filehandler)

filehandler = open("test_dump",'wb+')
pickle.dump(test,filehandler)

filehandler = open("user_dump",'wb+')
pickle.dump(users_map,filehandler)

filehandler = open("movie_dump",'wb+')
pickle.dump(movies_map,filehandler)
