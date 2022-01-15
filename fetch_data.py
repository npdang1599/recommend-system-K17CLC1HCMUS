from random import triangular
import pandas as pd
import numpy as np

#Input: mysql cursor
#Output: dataframe of clicks data
def rating_click_df(cur):
    cur.execute("""SELECT id_user, id_movie, is_clicked FROM moviedb.interactive""")
    res = cur.fetchall()
    click_df = pd.DataFrame(res, columns=['id_user', 'id_movie', 'rating'])
    return click_df

# Input: 
# + cur: mysql cursor
# + user_id: id of user
# Output:
# + dataframe of users and similarity level with input user
def similarity_df(cur, user_id):
    cur.execute("""SELECT id_user_2, similarity FROM moviedb.jaccard_similarity WHERE id_user_1 = %s""", (user_id,))
    res = cur.fetchall()
    similarity_df = pd.DataFrame(res, columns=['id_user_2', 'similarity'])
    return similarity_df

# Input: 
# + cur: mysql cursor
# + user_id: id of user
# Output:
# + np array of user factors with (1,k) shape
def user_factor(cur, user_id):
    cur.execute("""SELECT * FROM moviedb.user_factors WHERE id_user = %s""",(user_id,))
    res = cur.fetchall()
    user_factor = np.asarray(res, dtype= float).flatten()[1:]
    return user_factor

# Input: 
# + cur: mysql cursor
# Output:
# + np array of item factors with (n_item,k) shape
def item_factor(cur):
    cur.execute("""SELECT * FROM moviedb.movie_factors """)
    res = cur.fetchall()
    res = [ele[1:] for ele in res]
    movie_factor = np.asarray(res, dtype= float)
    return movie_factor

# Input: 
# + cur: mysql cursor
# + user_id: id of user
# Output:
# + user bias (float)
def user_bias(cur, user_id):
    cur.execute("""SELECT * FROM moviedb.user_biases WHERE id_user = %s""",(user_id,))
    res = cur.fetchall()
    user_bias = np.asarray(res, dtype= float).flatten()[1]
    return user_bias

# Input: 
# + cur: mysql cursor
# Output:
# + item bias in (n_item, 1) shape
def item_bias(cur):
    cur.execute("""SELECT * FROM moviedb.movie_biases""")
    res = cur.fetchall()
    res = [ele[1:] for ele in res]
    movie_bias = np.asarray(res, dtype= float).flatten()
    return movie_bias

# Input: 
# + cur: mysql cursor
# Output:
# + average global rating (float)
def global_rating_mean(cur ):
    cur.execute("""SELECT * FROM moviedb.global_mean_ratings """)
    res = cur.fetchall()
    global_mean_rating = np.asarray(res, dtype= float).flatten()[0]
    return global_mean_rating

#Input: mysql cursor
#Output: dataframe of watching time data
def rating_watchtime_df(cur):
    cur.execute("""SELECT id_user, id_movie, rating FROM moviedb.interactive""")
    res = cur.fetchall()
    training_df = pd.DataFrame(res, columns=['id_user', 'id_movie', 'rating'])
    training_df = training_df.dropna()
    return training_df

# Input: 
# + cur: mysql cursor
# + list_item_id: List of item ids
# Output:
# + dataframe contain id of movies and information of their directors and actors
def movie_director_actor(cur, list_item_id):
    cur.execute("""SELECT id, director, actor FROM moviedb.movie WHERE id IN %s""",(tuple(list_item_id),))
    # print('id: ', id)
    res = cur.fetchall()
    return pd.DataFrame(res, columns=['id','director','actor'])