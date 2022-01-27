from GroupMF_group import Group
import pandas as pd
import flask
from flask import request, jsonify
from flask_mysqldb import MySQL
from GroupMF_recommend_engine import RecSys
import time
import cold_start
from flask_cors import CORS
import KRNN_recommend_engine
import fetch_data
import utils

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

def individual_recommend_list_state1(id_user, n_movie):
    cur = mysql.connection.cursor()
    mov_ids = cold_start.get_movie_ids_from_db(cur,id_user)
    cur.close()

    cur = mysql.connection.cursor()
    if cold_start.check_new_user(mov_ids):
        print("New user detected!")
        rec_list = cold_start.get_recommend_list(mov_ids,n_movie,cur)
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

@app.route('/individual/state1/', methods=['GET'])
def individual_state1_api():
    if 'id_user' and 'n_movie' in request.args:
        id_user = int(request.args['id_user'])
        n_movie = int(request.args['n_movie'])

    else:
        return "Error: No id field provided. Please specify an id."
    
    results_with_sim, rec_list = individual_recommend_list_state1(id_user,n_movie)

    # results = pd.DataFrame(rec_list, columns=['id']).to_dict('records')

    result = utils.display_results(mysql, rec_list)
    # return jsonify(results)
    
    return jsonify(result)

@app.route('/group/state1/', methods=['GET'])
def group_recommend_list_state1():
    if 'id_user' and 'n_movie' in request.args:
        user_ids = request.args.getlist("id_user")
        n_movie = int(request.args['n_movie'])
    else:
        return "Error: No id field provided. Please specify an id."
    
    cur = mysql.connection.cursor()
    # training_df = fetch_data.rating_click_df(cur)
   
    df = pd.DataFrame()
    for i in range(len(user_ids)):
        # user_df = fetch_data.similarity_df(cur,i)
        
        # i_tmp, i_r_tmp = state1.recommend_sys(int(user_ids[i]), 10, training_df, user_df)
        rec_list_idv_df, rec_list_idv = individual_recommend_list_state1(int(user_ids[i]),n_movie)
        df = df.append([rec_list_idv_df])
    # print("df: ", df)
   
    g_items =  pd.DataFrame(df['Item'])
    g_items.drop_duplicates(subset ="Item", keep='first', inplace = True)
    result = pd.DataFrame()
    for index, row in g_items.iterrows():
        g_r = df[df['Item'] == row['Item']]
        new_gen = pd.DataFrame(g_r.groupby(['Item'], as_index=False)['Rating'].mean(),)
        new_gen.insert(0, 'User_count', len(g_r))
        result= result.append(new_gen, ignore_index=True)
    result=result.sort_values(['User_count','Rating'], ascending=[False, False])
    
    rec_list = result['Item'].to_list()[0:n_movie]
    
    cur.close()
    result = utils.display_results(mysql, rec_list)

    return jsonify(result)

def indv_state2_new_user(id_movie,n_movie):
    cur = mysql.connection.cursor()
    ratings = fetch_data.rating_watchtime_df(cur)
    cur.close()
    
    similar_ids = cold_start.find_similar_movies(id_movie, k=n_movie, ratings=ratings)
    return similar_ids

def indv_state2_old_user(id_user, n_movie):
    cur = mysql.connection.cursor()
    rec_sys = RecSys(cur)
    cur.close()

    user = [id_user]
    candidate_items = Group.find_candidate_items(rec_sys.ratings,user)
    idv = Group(user, candidate_items, rec_sys.ratings)

    cur = mysql.connection.cursor()
    rec_sys.idv_recommend(cur, idv)
    cur.close()

    rec_list = idv.reco_list[:n_movie]
    return rec_list

@app.route('/individual/state2/', methods=['GET']) # /idividual/state2?id=10
def individual_recommend_list_state2():
    if 'id_user' and 'id_movie' and 'n_movie' in request.args:
        id_user = int(request.args['id_user'])
        id_movie = int(request.args['id_movie'])
        n_movie = int(request.args['n_movie'])
    else:
        return "Error: No id field provided. Please specify an id."

    cur = mysql.connection.cursor()
    mov_ids = cold_start.get_movie_ids_from_db(cur,id_user)
    cur.close()

    if cold_start.check_new_user(mov_ids):
        print("New user detected!")
        rec_list = indv_state2_new_user(id_movie,n_movie)
    else:
        print("old user detected!")
        rec_list = indv_state2_old_user(id_user,n_movie)

    result = utils.display_results(mysql, rec_list)

    return jsonify(result)

@app.route('/group/state2/', methods=['GET'])
def group_recommend_list_state2():
    if 'id_user' and 'id_movie' and 'n_movie' in request.args:
            group_members = request.args.getlist("id_user")
            id_movie = int(request.args['id_movie'])
            n_movie = int(request.args['n_movie'])
    else:
        return "Error: No id field provided. Please specify an id."
    
    group_members = list(map(int, group_members))


    cur = mysql.connection.cursor()
    for id_user in group_members:
        print(id_user)

        mov_ids = cold_start.get_movie_ids_from_db(cur,id_user)
        if cold_start.check_new_user(mov_ids):
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