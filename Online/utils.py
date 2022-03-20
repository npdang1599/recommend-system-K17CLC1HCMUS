import numpy as np
import pandas as pd
import fetch_data

# convert_data_to_array: 
def to_utilize_matrix(training_data):
    num_users = max(training_data.id_user.unique())
    num_items = max(training_data.id_movie.unique())

    ratings = np.zeros((num_users, num_items))
    # test_ratings = np.zeros((num_users, num_items))

    for row in training_data.itertuples(index=False):
        ratings[row.id_user - 1, row.id_movie - 1] = row.rating
    
    return ratings

# find_candidate_items: find list of items that can be recommended.
# These should not have been watched by any member of group.
def find_candidate_items(ratings, members):
    if len(members) == 0: return []

    unwatched_items = np.argwhere(ratings[members[0]] == 0)
    for member in members:
        cur_unwatched = np.argwhere(ratings[member] == 0)
        unwatched_items = np.intersect1d(unwatched_items, cur_unwatched)

    return unwatched_items

# display_results: is used to config what show in json result
def display_results(mysql,list_item_id):
    # cur = mysql.connection.cursor()
    # cur.execute("""
    #             SELECT a.id, a.name, a.director, a.description, group_concat(c.name) AS genre
    #             FROM moviedb.movie a
    #             JOIN moviedb.movie_list b ON a.id = b.id_movie
    #             JOIN moviedb.list c ON c.id = b.id_list
    #             WHERE a.id IN %s AND c.type = 0
    #             GROUP BY a.id, a.name, a.director, a.description
    #             """,(tuple(list_item_id),))
    # # print('id: ', id)
    # res = cur.fetchall()
    # cur.close()
    # print("res: ", res)
    # item_id_df = pd.DataFrame(list_item_id, columns=['id'])
    # info_df = pd.DataFrame(res, columns=['id','movie_title','director','decription','gerne'])
    # item_id_df = item_id_df.join(info_df.set_index('id'), on='id', how='left')   
    # print('item_id_df: ', item_id_df)

    # return pd.DataFrame(res, columns=['id','movie_title','director','decription','gerne']).to_dict('records')
    # return item_id_df.to_dict('records')
    return pd.DataFrame(list_item_id, columns=['id']).to_dict('records')

# check_new_user: set threshold to determine wether the user is newuser or not
def check_new_user(cur, id_user):
    # get movies that had watched by input user from database
    # mov_ids = fetch_data.movie_watched_by_user(cur, id_user)

    # return mov_ids, len(mov_ids) < 50 
    return not fetch_data.is_old_user(cur,id_user)
    