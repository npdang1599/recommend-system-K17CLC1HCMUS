from group import Group
from config import Config

import numpy as np
import pandas as pd
import warnings

import flask
from flask_mysqldb import MySQL

# overflow warnings should be raised as errors
np.seterr(over='raise')

def average(arr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        arr[arr == 0] = np.nan
        return np.nanmean(arr, axis=0)

class RecSys:

    def __init__(self, mysql):
        self.cfg = Config(r"config.conf")
        
        # training and testing matrices
        self.ratings = None

        self.data_ready = 0   

        # output after svd factorization
        # initialize all unknowns with random values from -1 to 1
        # self.user_factors = None
        self.item_factors = None

        # self.user_biases = None
        self.item_biases = None

        # global mean of ratings a.k.a mu
        self.ratings_global_mean = None
        
        # read data into above matrices
        self.read_data(mysql)
        
        self.num_users = self.ratings.shape[0]
        self.num_items = self.ratings.shape[1]
        
        # predicted ratings matrix based on factors.
        self.predictions = np.zeros((self.num_users, self.num_items))
 

    def read_data(self, mysql):    
        cur = mysql.connection.cursor()

        cur.execute("""SELECT id_user, id_movie, rating FROM moviedb.interactive""")
        res = cur.fetchall()
        mysql.connection.commit()
        training_data = pd.DataFrame(res, columns=['id_user', 'id_movie', 'rating'])
        training_data = training_data.dropna()
        # print('training_data: ', training_data.head(25))
        # Change training_data dataframe to an array of data for calculation purposes
        num_users = max(training_data.id_user.unique())
        num_items = max(training_data.id_movie.unique())

        self.ratings = np.zeros((num_users, num_items))

        for row in training_data.itertuples(index=False):
            self.ratings[row.id_user - 1, row.id_movie - 1] = row.rating  
        
        cur.execute("""SELECT * FROM moviedb.movie_factors """)
        res = cur.fetchall()
        mysql.connection.commit()
        res = [ele[1:] for ele in res]
        self.item_factors = np.asarray(res, dtype= float)

        cur.execute("""SELECT * FROM moviedb.movie_biases""")
        res = cur.fetchall()
        mysql.connection.commit()
        res = [ele[1:] for ele in res]
        self.item_biases = np.asarray(res, dtype= float).flatten()

        cur.execute("""SELECT * FROM moviedb.global_mean_ratings """)
        res = cur.fetchall()
        mysql.connection.commit()
        self.ratings_global_mean = np.asarray(res, dtype= float).flatten()[0]

            
        cur.close()

# RecSys.read_data = read_data

def predict_group_rating(self, group, item):
    factors = group.grp_factors_bf; bias_group = group.bias_bf
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
    group.grp_factors_bf = factor_n_bias[:-1]
    group.bias_bf = factor_n_bias[-1]

    # Making recommendations on candidate list :
    group_candidate_ratings = {}
    for idx, item in enumerate(group.candidate_items):
        cur_rating = self.predict_group_rating(group, item-1)

        if (cur_rating > self.cfg.rating_threshold_bf):
            group_candidate_ratings[item-1] = cur_rating

    # sort and filter to keep top 'num_recos_bf' recommendations
    group_candidate_ratings = sorted(group_candidate_ratings.items(), key=lambda x: x[1], reverse=True)
    # [:self.cfg.num_recos_bf]

    group.reco_list_bf = np.array([rating_tuple[0] for rating_tuple in group_candidate_ratings])
RecSys.bf_runner = bf_runner

