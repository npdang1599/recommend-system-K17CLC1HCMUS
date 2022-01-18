from group import Group
from config import Config

import numpy as np
import pandas as pd
import warnings
import fetch_data


# overflow warnings should be raised as errors
np.seterr(over='raise')

def average(arr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        arr[arr == 0] = np.nan
        return np.nanmean(arr, axis=0)

class RecSys:

    def __init__(self, cur):
        self.cfg = Config(r"config.conf")
        
        # training and testing matrices
        self.ratings = None

        # output after svd factorization
        # initialize all unknowns with random values from -1 to 1
        # self.user_factors = None
        self.item_factors = None

        # self.user_biases = None
        self.item_biases = None
        

        # global mean of ratings a.k.a mu
        self.ratings_global_mean = None
        
        # read data into above matrices
        self.read_data(cur)
        
        self.num_users = self.ratings.shape[0]
        self.num_items = self.ratings.shape[1]
        
        # predicted ratings matrix based on factors.
        self.predictions = np.zeros((self.num_users, self.num_items))
 

def read_data(self, cur):    

    training_data = fetch_data.rating_watchtime_df(cur)

    # Change training_data dataframe to an array of data for calculation purposes
    num_users = max(training_data.id_user.unique())
    num_items = max(training_data.id_movie.unique())

    self.ratings = np.zeros((num_users, num_items))

    for row in training_data.itertuples(index=False):
        self.ratings[row.id_user - 1, row.id_movie - 1] = row.rating  
    
    self.item_factors = fetch_data.item_factor(cur)
    self.item_biases = fetch_data.item_bias(cur)
    self.ratings_global_mean = fetch_data.global_rating_mean(cur)
        
    cur.close()
RecSys.read_data = read_data

def predict_user_rating(self, user, item):
    prediction = self.ratings_global_mean + user.bias + self.item_biases[item]
    prediction += user.grp_factors.dot(self.item_factors[item, :].T)
    return prediction
RecSys.predict_user_rating = predict_user_rating

def idv_recommend(self,cur, user):

    print("user: ", user.members[0])
    user.grp_factors = fetch_data.user_factor(cur, user.members[0])
    user.bias = fetch_data.user_bias(cur, user.members[0])

    idv_candidate_ratings = {}
    for idx, item in enumerate(user.candidate_items):
        cur_rating = self.predict_user_rating(user, item-1)

        idv_candidate_ratings[item-1] = cur_rating

    # sort and filter to keep top 'num_recos_bf' recommendations
    idv_candidate_ratings = sorted(idv_candidate_ratings.items(), key=lambda x: x[1], reverse=True)
    # [:self.cfg.num_recos_bf]

    user.reco_list = np.array([rating_tuple[0] for rating_tuple in idv_candidate_ratings])
RecSys.idv_recommend = idv_recommend


def predict_group_rating(self, group, item):
    factors = group.grp_factors; bias_group = group.bias
    return self.ratings_global_mean + bias_group + self.item_biases[item] \
                                    + np.dot(factors.T, self.item_factors[item])
RecSys.predict_group_rating = predict_group_rating

def bf_runner(self, group):
    # aggregate user ratings into virtual group
    # calculate factors of group
    lamb = self.cfg.lambda_mf

    all_movies = np.arange(len(self.ratings.T))
    watched_items = sorted(list(set(all_movies) - set(group.candidate_items)))

    group_rating = self.ratings[group.members, :]
    agg_rating = average(group_rating)
    s_g = []
    for j in watched_items:
        s_g.append(agg_rating[j] - self.ratings_global_mean - self.item_biases[j])

    # creating matrix A : contains rows of [item_factors of items in watched_list + '1' vector]
    A = np.zeros((0, self.cfg.num_factors))

    for item in watched_items:
        A = np.vstack([A, self.item_factors[item]])
    v = np.ones((len(watched_items), 1))
    A = np.c_[A, v]

    factor_n_bias = np.dot(np.linalg.inv(np.dot(A.T, A) + lamb * np.identity(self.cfg.num_factors + 1)), np.dot(A.T, s_g))
    group.grp_factors = factor_n_bias[:-1]
    group.bias = factor_n_bias[-1]

    # Making recommendations on candidate list :
    group_candidate_ratings = {}
    for idx, item in enumerate(group.candidate_items):
        cur_rating = self.predict_group_rating(group, item-1)

        if (cur_rating > self.cfg.rating_threshold_bf):
            group_candidate_ratings[item-1] = cur_rating

    # sort and filter to keep top 'num_recos_bf' recommendations
    group_candidate_ratings = sorted(group_candidate_ratings.items(), key=lambda x: x[1], reverse=True)
    # [:self.cfg.num_recos_bf]

    group.reco_list = np.array([rating_tuple[0] for rating_tuple in group_candidate_ratings])
RecSys.bf_runner = bf_runner

