import pandas as pd
import numpy as np
from numpy.linalg import norm
import math 
from scipy import spatial
import copy

#Hàm tính Jaccard sim giữa hai người dùng
#Input: file training, 2 người dùng
#Output: Gía trị tương đồng giữa hai người dùng
def cal_sim(TrainingDataFrame, User1, User2):

    #Lấy danh sách item của 2 người dùng mà có đánh giá là 1 
    User1_items = TrainingDataFrame.Item[TrainingDataFrame.User == User1][TrainingDataFrame.Rating == 1].to_list()
    User2_items = TrainingDataFrame.Item[TrainingDataFrame.User == User2][TrainingDataFrame.Rating == 1].to_list()

    #Giao của 2 người dùng
    intersection = [value for value in User1_items if value in User2_items]

    #Chặn lại trong trường hợp không có item chung hay là mẫu bằng 0
    if len(intersection) == 0:
        return 0
    if (len(User1_items) + len(User2_items) - len(intersection)) == 0:
        return 0
    
    #Gía trị tương đồng Jaccard
    sim = len(intersection) / (len(User1_items) + len(User2_items) - len(intersection)) 
    
    return sim 
    
#Hàm trả về các giá trị tương đồng của người dùng được chọn với những người dùng còn lại
#Input: 
#TrainingDataFrame - bộ dữ liệu traning dưới dạng dataframe, User - id người dùng mục tiêu
#Output:
#Trả về một Dataframe chứa độ tương đồng của người dùng mục tiêu với những người dùng khác 
def sim_df(TrainingDataFrame, User):
    TrainingDataFrame.columns = ['User','Item','Rating']
    User_col =[]
    Sim_col = []
    #Vòng lặp hết người dùng
    for i in TrainingDataFrame.User.drop_duplicates():
        if User != i:
            User_col.append(i)
            Sim_col.append(cal_sim(TrainingDataFrame, User, i))
    
    #Tạo dataframe 
    df = pd.DataFrame(data={'User':User_col, 'Sim':Sim_col})

    #Sắp xếp độ tương đồng
    df = df.sort_values(by=['Sim'], axis=0,ascending=False, ignore_index=True)

    return df
