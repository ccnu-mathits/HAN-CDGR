import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import os

path = '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/MaFengWo/NATRbce/data/'

filename_gro = path + 'groupRatingTrain.txt'
filename_user = path + 'userRatingTrain.txt'
num_negatives = 4

def load_rating_file_as_matrix(filename):
    features_ps = []
    # Get number of users and items
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(" ")
            user, item = int(arr[0]), int(arr[1])
            num_users = max(num_users, user) 
            num_items = max(num_items, item) 
            features_ps.append([user, item])
            line = f.readline()
    # Construct matrix 
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(" ")
            if len(arr) > 2:
                user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
            else:
                user, item = int(arr[0]), int(arr[1])
                mat[user, item] = 1.0
            line = f.readline()
    return features_ps, num_users + 1, num_items + 1, mat

features_ps, num_users, num_items, train_mat = load_rating_file_as_matrix(filename_gro)

def get_train_instances(train_mat):
    user_input, pos_item_input, neg_item_input = [], [], []
    train_data_bce = []
    num_users = train_mat.shape[0]
    num_items = train_mat.shape[1]
    random.seed(1314)
    for (u, i) in train_mat.keys():
        train_data_bce.append([u,i,1])
        # positive instance
        for _ in range(num_negatives):
            pos_item_input.append(i)
        # negative instances
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train_mat:
                j = np.random.randint(num_items)
            user_input.append(u)
            neg_item_input.append(j)
            train_data_bce.append([u,j,0])

    train_data = np.array([user_input, pos_item_input, neg_item_input]).T        
    # pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
    train_data_bce = np.array(train_data_bce)
    return train_data, train_data_bce

train_data, train_data_bce = get_train_instances(train_mat)

np.savetxt(path + "groRatingTrainnew", train_data, fmt='%d', delimiter=' ')
np.savetxt(path + "groRatingTrainnew_bce", train_data_bce, fmt='%d', delimiter=' ')

print('Done!')

