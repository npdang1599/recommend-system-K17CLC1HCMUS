from random import triangular
import pandas as pd
import numpy as np
from numpy.linalg import norm
import math 
from scipy import spatial
import copy
import Jaccard_sim

# get_list_sim function: calculate similarity between user and store in a list
# Input;
# + rating_click_df: click data of user-movie interactive in the form of pandas dataframe with 3 cols (user_id, movie_id, click)
# Output:
# + list of Jaccard similarity scores dataframe 2 cols (user_2_id, sim)
def get_list_sim(rating_click_df, first_user, last_user):
    
    df_list=[]

    for i in range(first_user, last_user + 1, 1):
      print("user_get_list_sim: ", i)
      df = Jaccard_sim.sim_df(rating_click_df, i)
      df_list.append(df)
    
    return df_list

#Hàm tính toán lại sim của người dùng theo thuật toán K-RNN
#Input:
#df_list - list sim ở trên, gamma - hệ số nhân gamma, k là số láng giềng cho K-NN, path - đường dẫn lưu trữ cho sim mới
#first - người dùng bắt đầu vòng lặp, last - người dùng kết thúc vòng lặp 
#Output:
#file excel chứa độ tương đồng của các người dùng
def recal_sim(df_list, gamma, k, path, first, last):

  #Vòng lặp người dùng 
  for i in range(first - 1, last, 1):

    #In id của người dùng đang xét
    print("user_recal_sim: ",i + 1)

    #Lấy thông tính độ tương đồng và láng giềng của người dùng
    u_sim = df_list[i]
    u_neighbor = u_sim.User[0:k].tolist()

    #Vòng lặp thứ 2 so sánh từng cặp người dùng
    for j in u_sim.User:

      #Lấy độ tương đồng và láng giềng của người dùng w
      w_sim = df_list[j-1]
      w_neighbor = w_sim.User[0:k].tolist()

      #Xét điều kiện để tăng độ tương đồng
      if (i + 1) in w_neighbor and j in u_neighbor:
        #True - nằm trong tập láng giềng của nhau
        print("Increase sim between: ", i + 1, j)
        if (u_sim.Sim[u_sim.User == j] >= 0).bool() == True:
          #Nếu giá trị sim >= 0
          u_sim.loc[u_sim['User']==j, ['Sim']] *= (1 + gamma)
        else:
          #Nếu giá trị sim < 0
          u_sim.loc[u_sim['User']==j, ['Sim']] /= (1 + gamma)
      else:
        #False - ngược lại
        continue

    #Sắp xếp lại láng giềng theo giá trị sim mới
    u_sim = u_sim.sort_values(by=['Sim'], axis=0, ascending=False, ignore_index=True)

    #Chuyển sang file excel 
    u_sim.to_excel(r"{0}/{1}.xlsx".format(path, i + 1), index = False) 


