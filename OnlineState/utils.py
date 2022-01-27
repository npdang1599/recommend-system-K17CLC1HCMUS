import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

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

def display_results(mysql,list_item_id):

    return pd.DataFrame(list_item_id, columns=['id']).to_dict('records')
    # cur = mysql.connection.cursor()
    # cur.execute("""SELECT id, name, director, description FROM moviedb.movie WHERE id IN %s""",(tuple(list_item_id),))
    # # print('id: ', id)
    # res = cur.fetchall()
    # cur.close()
    # return pd.DataFrame(res, columns=['id','movie_title','director','decription']).to_dict('records')
    

def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.
    
    Args:
        df: pandas dataframe
    
    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    
    df.columns = ['userId','movieId','rating']
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper