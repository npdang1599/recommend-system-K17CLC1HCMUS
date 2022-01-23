import group
from group import Group
import numpy as np
import pandas as pd
import math
import flask
from flask import jsonify, request
from flask_mysqldb import MySQL
import time


def predict_user_rating(user_factor, item_factor, user_bias, item_bias, rating_global_mean):
    prediction = rating_global_mean + user_bias + item_bias
    prediction +=  user_factor.dot(item_factor.T)
    return prediction
  
def convert_data_to_array(training_data):
    num_users = max(training_data.id_user.unique())
    num_items = max(training_data.id_movie.unique())

    ratings = np.zeros((num_users, num_items))
    # test_ratings = np.zeros((num_users, num_items))

    for row in training_data.itertuples(index=False):
        ratings[row.id_user - 1, row.id_movie - 1] = row.rating
    
    return ratings

def find_candidate_items(ratings, members):
    if len(members) == 0: return []

    unwatched_items = np.argwhere(ratings[members[0]] == 0)
    for member in members:
        cur_unwatched = np.argwhere(ratings[member] == 0)
        unwatched_items = np.intersect1d(unwatched_items, cur_unwatched)

    return unwatched_items
# bảng này nữa, cái bảng thông tin movie
def displace_results(mysql,cur,list_item_id):
    cur.execute("""SELECT id, director, actor FROM moviedb.movie WHERE id IN %s""",(tuple(list_item_id),))
    # print('id: ', id)
    res = cur.fetchall()
    mysql.connection.commit()
    return pd.DataFrame(res, columns=['id','director','actor']).to_dict('records')
    



