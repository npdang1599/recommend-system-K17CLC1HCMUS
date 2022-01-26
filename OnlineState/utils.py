import numpy as np
import pandas as pd
def predict_user_rating(user_factor, item_factor, user_bias, item_bias, rating_global_mean):
    prediction = rating_global_mean + user_bias + item_bias
    prediction +=  user_factor.dot(item_factor.T)
    return prediction

# Attention!!!!
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

def display_results(mysql,list_item_id):

    return pd.DataFrame(list_item_id, columns=['id']).to_dict('records')
    # cur = mysql.connection.cursor()
    # cur.execute("""SELECT id, name, director, description FROM moviedb.movie WHERE id IN %s""",(tuple(list_item_id),))
    # # print('id: ', id)
    # res = cur.fetchall()
    # cur.close()
    # return pd.DataFrame(res, columns=['id','movie_title','director','decription']).to_dict('records')
    



