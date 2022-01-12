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

app = flask.Flask(__name__)
CORS(app)


app.config['MYSQL_HOST'] = '66.42.59.144'
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_USER'] = 'lucifer'
app.config['MYSQL_PASSWORD'] = '123123'
app.config['MYSQL_DB'] = 'moviedb'

mysql = MySQL(app)


@app.route('/', methods=['GET'])
def home():
    return """<h1>Movie recommend engine</h1>
              <p>This site is APIs for getting list of recommend movies.</p>"""
    
@app.route('/individual/state1/', methods=['GET'])
def individual_recommend_list_state1():
    if 'id' in request.args:
        id_user = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."
    
    # start = time.time()
    cur = mysql.connection.cursor()
    
    cur.execute("""SELECT id_user, id_movie, is_clicked FROM moviedb.interactive""")
    res = cur.fetchall()
    mysql.connection.commit()
    # end = time.time()
    # print("elapse time: ", end-start)
    training_df = pd.DataFrame(res, columns=['id_user', 'id_movie', 'rating'])

    
    # start = time.time()
    cur.execute("""SELECT id_user_2, similarity FROM moviedb.jaccard_similarity WHERE id_user_1 = %s""", (id_user,))
    res = cur.fetchall()
    mysql.connection.commit()
    # end = time.time()
    # print("elapse time: ", end-start)
    user_df = pd.DataFrame(res, columns=['id_user_2', 'similarity'])
    
    # results = []
    # start = time.time()
    rec_list, df = utils.recommend_list(id_user, 10, training_df, user_df)
    # results = df.to_dict('record')
    # end = time.time()
    # print("elapse time: ", end-start)

    rec_list =  rec_list[0:10]
    results = pd.DataFrame(rec_list, columns=['id']).to_dict('records')
    cur.close()

    return jsonify(results)

@app.route('/group/state1/', methods=['GET'])
def group_recommend_list_state1():
    if 'id' in request.args:
        user_ids = request.args.getlist("id")
    else:
        return "Error: No id field provided. Please specify an id."
    
    cur = mysql.connection.cursor()
 
    cur.execute("""SELECT id_user, id_movie, is_clicked FROM moviedb.interactive""")
    res = cur.fetchall()
    mysql.connection.commit()
    # end = time.time()
    # print("elapse time: ", end-start)
    training_df = pd.DataFrame(res, columns=['id_user', 'id_movie', 'rating'])
    
   
    df = pd.DataFrame()
    for i in range(len(user_ids)):
        # print('i:', i)
        # print('user_',user_ids[i])
        
        cur.execute("""SELECT id_user_2, similarity FROM moviedb.jaccard_similarity WHERE id_user_1 = %s""", (i,))
        res = cur.fetchall()
        mysql.connection.commit()
        user_df = pd.DataFrame(res, columns=['id_user_2', 'similarity'])
        
        i_tmp, i_r_tmp = utils.recommend_list(int(user_ids[i]), 10, training_df, user_df)
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

@app.route('/individual/state2/', methods=['GET'])
def individual_recommend_list_state2():
    if 'id' in request.args:
        id_user = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."
    cur = mysql.connection.cursor()
    
    # Get data of user factor:
    cur.execute("""SELECT * FROM moviedb.user_factors WHERE id_user = %s""",(id_user,))
    res = cur.fetchall()
    mysql.connection.commit()
    user_factor = np.asarray(res, dtype= float).flatten()[1:]
    # print('user_factor: ', user_factor,'\nType: ',type(user_factor),'\nShape: ', user_factor.shape,'\n')
   
    # Get data of item factors
    cur.execute("""SELECT * FROM moviedb.movie_factors """)
    res = cur.fetchall()
    mysql.connection.commit()
    res = [ele[1:] for ele in res]
    movie_factor = np.asarray(res, dtype= float)
    # print('item_factor: ',movie_factor,'\nType: ',type(movie_factor),'\nShape: ', movie_factor.shape,'\n')
   
    # Get data of user bias
    # Chỗ này có id_user k?
    # sao nó nói k thế
    cur.execute("""SELECT * FROM moviedb.user_biases WHERE id_user = %s""",(id_user,))
    res = cur.fetchall()
    mysql.connection.commit()
    user_bias = np.asarray(res, dtype= float).flatten()[1]
    # print('user_bias: ', user_bias,'\nType: ',type(user_bias),'\nShape: ', user_bias.shape,'\n')
    
   
    # Get data of movie bias
    cur.execute("""SELECT * FROM moviedb.movie_biases""")
    res = cur.fetchall()
    mysql.connection.commit()
    res = [ele[1:] for ele in res]
    movie_bias = np.asarray(res, dtype= float).flatten()
    # print('movie_bias: ', movie_bias,'\nType: ',type(movie_bias),'\nShape: ', movie_bias.shape,'\n')
    
   
    # Get data of global rating mean:
    cur.execute("""SELECT * FROM moviedb.global_mean_ratings """)
    res = cur.fetchall()
    mysql.connection.commit()
    global_mean_rating = np.asarray(res, dtype= float).flatten()[0]
    # print('global_mean_rating: ', global_mean_rating,'\nType: ',type(global_mean_rating),'\nShape: ', global_mean_rating.shape,'\n')
    
   
    # Get rating datas:
    cur.execute("""SELECT id_user, id_movie, rating FROM moviedb.interactive""")
    res = cur.fetchall()
    mysql.connection.commit()
    training_df = pd.DataFrame(res, columns=['id_user', 'id_movie', 'rating'])
    # print('training_data: ', training_df.head(25))
    training_df=training_df.dropna()
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


@app.route('/individual/new_user/', methods=['GET'])
def new_user_recommend_list():
    if 'id' in request.args:
        movie_ids = request.args.getlist("id")
    else:
        return "Error: No id field provided. Please specify an id."
    movie_ids = [int(item) for item in movie_ids]
    print(type(movie_ids))
    print(movie_ids)
    
    features = cold_start.get_genre(mysql)
    cosine_sim = cold_start.cosine_sim(features)
    res = cold_start.get_content_based_recommendations(movie_ids, cosine_sim)
    print("res: ", res)
    print("type: ", type(res))
    
    results = pd.DataFrame(res, columns=['id']).to_dict('records')
    

    return jsonify(results)
    
app.run()