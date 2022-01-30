import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import fetch_data

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
    return pd.DataFrame(list_item_id, columns=['id']).to_dict('records')
    # cur = mysql.connection.cursor()
    # cur.execute("""SELECT id, name, director, description FROM moviedb.movie WHERE id IN %s""",(tuple(list_item_id),))
    # # print('id: ', id)
    # res = cur.fetchall()
    # cur.close()
    # return pd.DataFrame(res, columns=['id','movie_title','director','decription']).to_dict('records')
    

# check_new_user: set threshold to determine wether the user is newuser or not
def check_new_user(cur, id_user):
    # get movies that had watched by input user from database
    mov_ids = fetch_data.movie_watched_by_user(cur, id_user)
    return mov_ids, len(mov_ids) < 20