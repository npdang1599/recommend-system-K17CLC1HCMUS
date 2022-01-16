import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import group
from aggregator import Aggregators
import fetch_data
import MySQLdb

# overflow warnings should be raised as errors
np.seterr(over='raise')

class RecSys:
    def __init__(self):
        self.cfg = group.Config(r"./config/config.conf")
        
        # training and testing matrices
        self.ratings = None
        
        # items information
        self.items_info = None

        # read data into above matrices
        self.read_data()
        
        self.num_users = self.ratings.shape[0]
        self.num_items = self.ratings.shape[1]
        
        # predicted ratings matrix based on factors.
        self.predictions = np.zeros((self.num_users, self.num_items))
        
        # output after svd factorization
        # initialize all unknowns with random values from -1 to 1
        self.user_factors = np.random.uniform(-1, 1, (self.ratings.shape[0], self.cfg.num_factors))
        self.item_factors = np.random.uniform(-1, 1, (self.ratings.shape[1], self.cfg.num_factors))

        self.user_biases = np.zeros(self.num_users)
        self.item_biases = np.zeros(self.num_items)
        
        # global mean of ratings a.k.a mu
        self.ratings_global_mean = 0

# read training and testing data into matrices
def read_data(self):
    # column_headers = ['user_id', 'item_id', 'rating', 'timestamp']

    # print('Reading training data from ', self.cfg.training_file, '...')
    # viewtime_data = pd.read_csv(self.cfg.training_file, sep='\t', names=column_headers) 
    conn = MySQLdb.connect(host="66.42.59.144", user="lucifer", passwd="12344321", db="moviedb")
    cur = conn.cursor()
    viewtime_data = fetch_data.rating_watchtime_df(cur)
    viewtime_data.columns=['user_id', 'item_id', 'rating']
    viewtime_data.dropna()
    cur.close()

    num_users = max(viewtime_data.user_id.unique())
    num_items = max(viewtime_data.item_id.unique())

    self.ratings = np.zeros((num_users, num_items))
    # self.test_ratings = np.zeros((num_users, num_items))

    for row in viewtime_data.itertuples(index=False):
        self.ratings[row.user_id - 1, row.item_id - 1] = row.rating
RecSys.read_data = read_data

def sgd_factorize(self):
    #solve for these for matrix ratings        
    ratings_row, ratings_col = self.ratings.nonzero()
    num_ratings = len(ratings_row)
    learning_rate = self.cfg.learning_rate_mf
    regularization = self.cfg.lambda_mf

    self.ratings_global_mean = np.mean(self.ratings[np.where(self.ratings != 0)])

    print('Doing matrix factorization...')
    try:
        for iter in range(self.cfg.max_iterations_mf):
            print('Iteration: ', iter)
            rating_indices = np.arange(num_ratings)
            np.random.shuffle(rating_indices)

            for idx in rating_indices:
                user = ratings_row[idx]
                item = ratings_col[idx]

                pred = self.predict_user_rating(user, item)
                error = self.ratings[user][item] - pred

                self.user_factors[user] += learning_rate \
                                            * ((error * self.item_factors[item]) - (regularization * self.user_factors[user]))
                self.item_factors[item] += learning_rate \
                                            * ((error * self.user_factors[user]) - (regularization * self.item_factors[item]))

                self.user_biases[user] += learning_rate * (error - regularization * self.user_biases[user])
                self.item_biases[item] += learning_rate * (error - regularization * self.item_biases[item])

            self.sgd_mse()

    except FloatingPointError:
        print('Floating point Error: ')
RecSys.sgd_factorize = sgd_factorize

def sgd_mse(self):
    self.predict_all_ratings()
    predicted_training_ratings = self.predictions[self.ratings.nonzero()].flatten()
    actual_training_ratings = self.ratings[self.ratings.nonzero()].flatten()

    training_mse = mean_squared_error(predicted_training_ratings, actual_training_ratings)
    print('training mse: ', training_mse)

RecSys.sgd_mse = sgd_mse

def predict_user_rating(self, user, item):
    prediction = self.ratings_global_mean + self.user_biases[user] + self.item_biases[item]
    prediction += self.user_factors[user, :].dot(self.item_factors[item, :].T)
    return prediction
RecSys.predict_user_rating = predict_user_rating

def predict_group_rating(self, group, item):
    factors = group.grp_factors_bf; bias_group = group.bias_bf
    return self.ratings_global_mean + bias_group + self.item_biases[item] \
                                    + np.dot(factors.T, self.item_factors[item])
RecSys.predict_group_rating = predict_group_rating

def predict_all_ratings(self):
    for user in range(self.num_users):
        for item in range(self.num_items):
            self.predictions[user, item] = self.predict_user_rating(user, item)
RecSys.predict_all_ratings = predict_all_ratings

def bf_runner(self, group, aggregator=Aggregators.average_bf):
    # aggregate user ratings into virtual group
    # calculate factors of group
    lamb = self.cfg.lambda_mf

    all_movies = np.arange(len(self.ratings.T))
    watched_items = sorted(list(set(all_movies) - set(group.candidate_items)))

    group_rating = self.ratings[group.members, :]
    agg_rating = aggregator(group_rating)
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
        cur_rating = self.predict_group_rating(group, item)

        if (cur_rating > self.cfg.rating_threshold_bf):
            group_candidate_ratings[item] = cur_rating

    # sort and filter to keep top 'num_recos_bf' recommendations
    group_candidate_ratings = sorted(group_candidate_ratings.items(), key=lambda x: x[1], reverse=True)
    # [:self.cfg.num_recos_bf]

    group.reco_list_bf = np.array([rating_tuple[0] for rating_tuple in group_candidate_ratings])
        
RecSys.bf_runner = bf_runner