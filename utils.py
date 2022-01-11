import group
from group import Group
import numpy as np
import pandas as pd
import math
import flask
from flask import jsonify, request
from flask_mysqldb import MySQL
import time


#đọc file training
# training_df = pd.read_excel(r"E:\Study\FinalProject\MBRecommend\data\training.xlsx")

#Hàm trả về danh sách item tư vấn cho người dùng
#Input: 
#user - id người dùng đang xét, N - số lượng item, training_df - file training, sim_path - đường dẫn thư mục độ tương đồng
#Output:
#Danh sách N item tư vấn cho người dùng
def recommend_list(user, N, training_df, user_df):
    item_col=[]
    rating_col=[]

    # start = time.time()
    #Lấy ra độ tương đồng của người dùng 
    #user_df = pd.read_excel(r"{0}/{1}.xlsx".format(sim_path, user))
    ratings = convert_data_to_array(training_df)
    #các item của người dùng có đánh giá tính cực
    #user_item = training_df.id_movie[training_df.id_user == user][training_df.rating == 1].to_list()

    #danh sách các item cần dự đoán
    #predict_list = [value for value in range(1,3953) if value not in user_item]
    predict_list = find_candidate_items(ratings, [user])
    # end = time.time()
    # print("elapse time 1: ", end-start)
    
    # start = time.time()
    #vòng lặp item
    for j in predict_list:
        #Danh sách những người dùng có đánh giá item j  
        user_j = training_df.id_user[training_df.id_movie == j].to_list()

        #Tổng sim 10 người dùng (l = 10) có đánh giá tới item j và có độ tương đồng lớn nhất
        sum_sim = sum(user_df.similarity[user_df.id_user_2.isin(user_j) == True][:10].to_list())

        #thêm id item j vào cột item
        item_col.append(j)

        #Thêm sim vào cột đánh giá
        rating_col.append(sum_sim)
        
    # end = time.time()
    # print("elapse time 1: ", end-start)

    #Tạo dataframe từ cột item và cột đánh giá
    df = pd.DataFrame(data={'Item':item_col,'Rating':rating_col})

    #Sắp xếp đánh giá dự đoán theo giá trị đánh giá
    df = df.sort_values(by=['Rating'],axis=0,ascending=False, ignore_index=True)

    #Lấy danh sách item từ 1 tới N
    df =  df.iloc[:N]

    #Trả về danh sách N item 
    return df['Item'].to_list(), df

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
    



