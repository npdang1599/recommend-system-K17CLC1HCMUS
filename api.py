from OfflineState.MF.fetch_data import user_factor
import group
from group import Group
import numpy as np
import pandas as pd
import math
import flask
from flask import request, jsonify
from flask_mysqldb import MySQL
import utils
from recommend_engine import RecSys
import time
import cold_start
from flask_cors import CORS
import state1
import fetch_data

app = flask.Flask(__name__)
CORS(app)

app.config['MYSQL_HOST'] = '66.42.59.144'
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_USER'] = 'lucifer'
app.config['MYSQL_PASSWORD'] = '12344321'
app.config['MYSQL_DB'] = 'moviedb'

mysql = MySQL(app)

@app.route('/', methods=['GET'])
def home():

    ratings = cold_start.get_rating_data_from_db(mysql)
    similar_ids = cold_start.find_similar_movies(1, k=20, ratings=ratings)

    results = pd.DataFrame(similar_ids, columns=['id']).to_dict('records')
    # return """<h1>Movie recommend engine</h1>
    #           <p>This site is APIs for getting list of recommend movies.</p>"""
    return jsonify(results)

@app.route('/individual/state1/', methods=['GET'])
def individual_recommend_list_state1():
    if 'id' in request.args:
        id_user = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."
    
    mov_ids = cold_start.get_movie_ids_from_db(mysql,id_user)
    if cold_start.check_new_user(mov_ids):
        print("New user detected!")
        rec_list = cold_start.get_recommend_list(mov_ids,10,mysql)
    else:
        print("Old user detected!")

        # start = time.time()
        #training, user = state1.fetch_data(mysql, id_user)
        cur = mysql.connection.cursor()
        click_df = fetch_data.rating_click_df(cur)
        sim_df = fetch_data.similarity_df(cur,id_user)
        cur.close()
        # end = time.time()
        # print("elapse time: ", end-start)

        # start = time.time()
        rec_list = state1.recommend_sys(id_user, 10, click_df, sim_df)
        # end = time.time()
        # print("elapse time: ", end-start)

    results = pd.DataFrame(rec_list, columns=['id']).to_dict('records')

    return jsonify(results)

@app.route('/group/state1/', methods=['GET'])
def group_recommend_list_state1():
    if 'id' in request.args:
        user_ids = request.args.getlist("id")
    else:
        return "Error: No id field provided. Please specify an id."
    
    cur = mysql.connection.cursor()
    training_df = fetch_data.rating_click_df(cur)
   
    df = pd.DataFrame()
    for i in range(len(user_ids)):
        user_df = fetch_data.similarity_df(cur,i)
        
        i_tmp, i_r_tmp = state1.recommend_sys(int(user_ids[i]), 10, training_df, user_df)
        df = df.append([i_r_tmp])
   
    g_items =  pd.DataFrame(df['Item'])
    g_items.drop_duplicates(subset ="Item", keep='first', inplace = True)
    result = pd.DataFrame()
    for index, row in g_items.iterrows():
        g_r = df[df['Item'] == row['Item']]
        new_gen = pd.DataFrame(g_r.groupby(['Item'], as_index=False)['Rating'].mean(),)
        new_gen.insert(0, 'User_count', len(g_r))
        result= result.append(new_gen, ignore_index=True)
    result=result.sort_values(['User_count','Rating'], ascending=[False, False])
    
    rec_list = result['Item'].to_list()[0:10]
    results = pd.DataFrame(rec_list, columns=['id']).to_dict('records')
    cur.close()

    return jsonify(results)

@app.route('/individual/state2/', methods=['GET']) # /idividual/state2?id=10
def individual_recommend_list_state2():
    if 'id' in request.args:
        id_user = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."
    cur = mysql.connection.cursor()
    
    # Get data of user factor:
    user_factor = fetch_data.user_factor(cur, id_user)

    # Get data of item factors
    movie_factor = fetch_data.item_factor(cur)
   
    # Get data of user bias
    user_bias = fetch_data.user_bias(cur, id_user)
    
    # Get data of movie bias
    movie_bias = fetch_data.item_bias(cur)
    
    # Get data of global rating mean:
    global_mean_rating = fetch_data.global_rating_mean(cur)
    
    # Get rating datas:
    training_df = fetch_data.rating_watchtime_df(cur)
    ratings = utils.convert_data_to_array(training_df)
   
    # Get candidate items:
    candidate_movie = utils.find_candidate_items(ratings, [id_user])
    # print("candidate: ", candidate_movie.shape)

    # Get recommend list:
    group_candidate_ratings = {}
    for idx, item in enumerate(candidate_movie):
        cur_rating = utils.predict_user_rating(user_factor, movie_factor[item-1], user_bias, movie_bias[item-1], global_mean_rating)

        group_candidate_ratings[item] = cur_rating

    group_candidate_ratings = sorted(group_candidate_ratings.items(), key=lambda x: x[1], reverse=True)

    rec_list = np.array([rating_tuple[0] for rating_tuple in group_candidate_ratings])
    

    rec_list = rec_list[:10]
    # print('result: ', result[:10])
    # result = [2,6]
    results = pd.DataFrame(rec_list, columns=['id']).to_dict('records')
    cur.close()

    return jsonify(results)

@app.route('/group/state2/', methods=['GET'])
def group_recommend_list_state2():
    if 'id' in request.args:
            group_members = request.args.getlist("id")
    else:
        return "Error: No id field provided. Please specify an id."
    
    rec_sys = RecSys(mysql)

    group_members = list(map(int, group_members))
    candidate_items = Group.find_candidate_items(rec_sys.ratings, group_members)
    gr = Group(group_members, candidate_items, rec_sys.ratings)
    rec_sys.bf_runner(gr)
    # print('Group recommended list:', gr.reco_list_bf[:10])
    
    cur = mysql.connection.cursor()
    rec_list = gr.reco_list_bf[:10]
    results = pd.DataFrame(rec_list, columns=['id']).to_dict('records')
    cur.close()

    return jsonify(results)
app.run()