from operator import is_
from numpy import rec
from GroupMF_group import Group
import pandas as pd
import flask
from flask import request, jsonify
from flask_mysqldb import MySQL
from GroupMF_recommend_engine import RecSys
import time
import cold_start_KNN_genre
import cold_start_KNN_time_watched
from flask_cors import CORS
import KRNN_recommend_engine
import fetch_data
import utils
import json

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
    return """<h1>Movie recommend engine</h1>
              <p>This site is APIs for getting list of recommend movies.</p>"""

# Recommend list for individual user (new user included)
def individual_recommend_list_state1(id_user, n_movie):
    # cur = mysql.connection.cursor()
    # mov_ids = fetch_data.movie_watched_by_user(cur,id_user)
    # cur.close()

    cur = mysql.connection.cursor()
    mov_ids, is_newuser = utils.check_new_user(cur, id_user)
    if is_newuser:
        print("New user detected!")
        rec_list, rec_list_w_score = cold_start_KNN_genre.get_recommend_list(mov_ids,n_movie,cur)
        rec_df = pd.DataFrame({'Item':rec_list})
        rec_df['Rating'] = 0
        cur.close()
    else:
        print("Old user detected!")
        click_df = fetch_data.rating_click_df(cur)
        sim_df = fetch_data.similarity_df(cur,id_user)
        cur.close()

        rec_df, rec_list = KRNN_recommend_engine.recommend_sys(id_user, n_movie, click_df, sim_df)
    
    return rec_df, rec_list


# Individual state 1 api: cold-start algorithm for new users and KRNN algorithm for old users
@app.route('/individual/state1/', methods=['GET'])
def individual_state1_api():
    if 'id_user' and 'n_movie' in request.args:
        id_user = int(request.args['id_user'])
        n_movie = int(request.args['n_movie'])
    else:
        return """Error: No id field provided. Please specify an id.
                (URL: /individual/state1?id_user= ... &n_movie= ...)
                """
    
    results_with_sim, rec_list = individual_recommend_list_state1(id_user,n_movie)
    result = utils.display_results(mysql, rec_list)
    
    return jsonify(result)

# Group state 1 api: join individual recommend results from Individual state 1's algorithms
@app.route('/group/state1/', methods=['GET'])
def group_recommend_list_state1():
    if 'id_user' and 'n_movie' in request.args:
        user_ids = request.args.getlist("id_user")
        n_movie = int(request.args['n_movie'])
    else:
        return """Error: No id field provided. Please specify an id.
                (/group/state1?id_user= ... & id_user= ... & ... &n_movie= ...)
                """
     
    cur = mysql.connection.cursor()
    # Perform for loop to get results from individual recommend algorithms
    agg_df = pd.DataFrame()
    for i in range(len(user_ids)):
        rec_list_idv_df, rec_list_idv = individual_recommend_list_state1(int(user_ids[i]),n_movie)
        agg_df = agg_df.append([rec_list_idv_df])
    agg_df.columns = ['Item','Rating']
    # agg_df: Dataframe that is aggregated from indvidual recommend results (two cols: Item and Rating)

    # Join results: 
    # + First criteria: the more the items appear in recommend lists of users, the higher pos they will appear in the final result
    # + Second: the higher Rating score of the item, the higher pos
    g_items =  pd.DataFrame(agg_df['Item'])
    g_items.drop_duplicates(subset ="Item", keep='first', inplace = True)
    
    result = pd.DataFrame()
    for index, row in g_items.iterrows():
        g_r = agg_df[agg_df['Item'] == row['Item']]
        new_gen = pd.DataFrame(g_r.groupby(['Item'], as_index=False)['Rating'].mean(),)
        new_gen.insert(0, 'User_count', len(g_r))
        result= result.append(new_gen, ignore_index=True)
    
    result=result.sort_values(['User_count','Rating'], ascending=[False, False])
    rec_list = result['Item'].to_list()[0:n_movie]
    cur.close()
    result = utils.display_results(mysql, rec_list)

    return jsonify(result)

def indv_state2_new_user(id_movie,n_movie,filter_by='default'):
    
    cur = mysql.connection.cursor()
    if filter_by == 'description':
        des_sim_df = fetch_data.description_similarity(cur,id_movie)
        cur.close()
        # print(des_sim_df)
        similar_ids = des_sim_df['id'].head(n_movie)
    elif filter_by == 'genre':
        rec_list,rec_list_w_score = cold_start_KNN_genre.get_recommend_list([id_movie],n_movie,cur)
        des_sim_df = pd.DataFrame(rec_list_w_score, columns=['id','similarity'])
        cur.close()
        # print(des_sim_df)
        similar_ids = des_sim_df['id'].head(n_movie)
    else:
        ratings_df = fetch_data.rating_watchtime_df(cur)
        cur.close() 
        similar_ids = cold_start_KNN_time_watched.find_similar_movies(id_movie, k=n_movie, ratings=ratings_df)

    return similar_ids

def indv_state2_old_user(id_user, n_movie, id_movie, filter_by='default'):
    cur = mysql.connection.cursor()
    rec_sys = RecSys(cur)
    cur.close()

    user = [id_user]
    candidate_items = Group.find_candidate_items(rec_sys.ratings,[id_user-1])
    idv = Group(user, candidate_items, rec_sys.ratings)

    cur = mysql.connection.cursor()
    rec_sys.idv_recommend(cur, idv)
    cur.close()

    # rec_list = idv.reco_list[:n_movie]
    if filter_by == 'description':
        rec_df = pd.DataFrame(idv.reco_list, columns=['id', 'rating'])
        cur = mysql.connection.cursor()
        rec_df = filter_by_description_sim(cur, id_movie, n_movie, rec_df)
        cur.close()
    elif filter_by == 'genre':
        rec_df = pd.DataFrame(idv.reco_list, columns=['id', 'rating'])
        cur = mysql.connection.cursor()
        rec_df = filter_by_genre(cur, id_movie, n_movie, rec_df)
        cur.close()
    else:
        rec_df = pd.DataFrame(idv.reco_list[:n_movie], columns=['id', 'rating'])

    per_match = round((rec_sys.predict_user_rating(idv,id_movie-1)/5)*100,0)

    return per_match, rec_df

def filter_by_description_sim(cur,movie_id, n_movie, predict_rating_df):
    des_sim_df = fetch_data.description_similarity(cur, movie_id)
    df_inner = pd.merge(predict_rating_df, des_sim_df, on='id', how='inner')
    df_inner['score']=df_inner['rating']*df_inner['similarity']
    df_inner = df_inner.sort_values(by='score', ascending= False)
    # print(df_inner)
    df_inner = df_inner.drop(['similarity','score'],axis = 1)
    return df_inner.head(n_movie)

def filter_by_genre(cur,movie_id, n_movie, predict_rating_df):
    rec_list,rec_list_w_score = cold_start_KNN_genre.get_recommend_list([movie_id],n_movie,cur)
    # print('rec_list_w_score ',rec_list_w_score)
    des_sim_df = pd.DataFrame(rec_list_w_score, columns=['id','similarity'])
    df_inner = pd.merge(predict_rating_df, des_sim_df, on='id', how='inner')
    df_inner['score']=df_inner['rating']*df_inner['similarity']
    df_inner = df_inner.sort_values(by='score', ascending= False)
    # print(df_inner)
    df_inner = df_inner.drop(['similarity','score'],axis = 1)
    return df_inner.head(n_movie)

@app.route('/individual/state2/', methods=['GET'])
def individual_recommend_list_state2():
    if 'id_user' and 'id_movie' and 'n_movie' and 'filter' in request.args:
        id_user = int(request.args['id_user'])
        id_movie = int(request.args['id_movie'])
        n_movie = int(request.args['n_movie'])
        filter = request.args['filter']
    else:
        return """Error: No id field provided. Please specify an id.
                (URL: /individual/state2?id_user= ... &n_movie= ... &id_movie= ... &filter= (description, genre, default))
                """
    cur = mysql.connection.cursor()
    mov_ids, is_newuser = utils.check_new_user(cur, id_user)
    cur.close()

    per_match = 0
    if is_newuser:
        print("New user detected!")
        rec_list = indv_state2_new_user(id_movie,n_movie,filter)
        result = pd.DataFrame(rec_list, columns=['id'])
        result['percentage_match'] = 0
        result = result.to_dict('records')
    else:
        print("old user detected!")
        per_match, rec_df = indv_state2_old_user(id_user,n_movie, id_movie,filter)

        rec_df.columns=['id', 'percentage_match']
        rec_df['percentage_match'] = rec_df['percentage_match'].apply(lambda x: round((x/5)*100,0))
        result = rec_df.to_dict('records')
    res ={
        'percentage_match': per_match,
        'list_recommend':result
    }   

    return jsonify(res)

@app.route('/group/state2/', methods=['GET'])
def group_recommend_list_state2():
    if 'id_user' and 'id_movie' and 'n_movie' in request.args:
            group_members = request.args.getlist("id_user")
            id_movie = int(request.args['id_movie'])
            n_movie = int(request.args['n_movie'])
    else:
        return """Error: No id field provided. Please specify an id.
                (URL: /group/state2?id_user= ... & id_user= ... & ... &n_movie= ... &id_movie= ...)
                """
    group_members = list(map(int, group_members))

    cur = mysql.connection.cursor()
    for id_user in group_members:
        # print(id_user)

        mov_ids, is_newuser = utils.check_new_user(cur, id_user)
        if is_newuser:
            print("New user detected!")
            group_members.remove(id_user)
    cur.close()

    if len(group_members) == 0:
        rec_list = indv_state2_new_user(id_movie, n_movie)
    else: 
        cur = mysql.connection.cursor()
        rec_sys = RecSys(cur)
        cur.close()

        group_members =[member - 1 for member in group_members]
        candidate_items = Group.find_candidate_items(rec_sys.ratings, group_members)
        gr = Group(group_members, candidate_items, rec_sys.ratings)
        
        rec_sys.bf_runner(gr)
        rec_list = gr.reco_list[:n_movie]
    
    result = utils.display_results(mysql, rec_list)
    return jsonify(result)
app.run()