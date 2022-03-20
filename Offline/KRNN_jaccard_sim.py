import MySQLdb
from alive_progress import alive_bar
import Jaccard_sim

# get_list_sim function: calculate similarity between user and store in a list
# Input;
# + rating_click_df: click data of user-movie interactive in the form of pandas dataframe with 3 cols (user_id, movie_id, click)
# Output:
# + list of Jaccard similarity scores dataframe 2 cols (user_2_id, sim)y
def get_list_sim(rating_click_df, user_lst):
  df_dict = {}
  with alive_bar(len(user_lst)) as bar:
    # for i in range(first_user, last_user + 1, 1):
    for i in user_lst:
      # print("user_get_list_sim: ", i)
      df = Jaccard_sim.sim_df(rating_click_df, i)
      tmp = {i:df}
      df_dict.update(tmp)
      bar()

  return df_dict

#Hàm tính toán lại sim của người dùng theo thuật toán K-RNN
#Input:
#df_list - list sim ở trên, gamma - hệ số nhân gamma, k là số láng giềng cho K-NN, path - đường dẫn lưu trữ cho sim mới
#first - người dùng bắt đầu vòng lặp, last - người dùng kết thúc vòng lặp 
#Output:
#file excel chứa độ tương đồng của các người dùng
def recal_sim(df_list, gamma, k, user_lst):
  #Vòng lặp người dùng 
  with alive_bar(len(user_lst)) as bar:
    for i in user_lst:

      #In id của người dùng đang xét
      # print("user_recal_sim: ",i)

      #Lấy thông tin độ tương đồng và láng giềng của người dùng
      u_sim = df_list[i]
      u_neighbor = u_sim.User[0:k].tolist()

      #Vòng lặp thứ 2 so sánh từng cặp người dùng
      for j in u_sim.User:

        #Lấy độ tương đồng và láng giềng của người dùng w
        w_sim = df_list[j]
        w_neighbor = w_sim.User[0:k].tolist()

        #Xét điều kiện để tăng độ tương đồng
        if (i + 1) in w_neighbor and j in u_neighbor:
        #True - nằm trong tập láng giềng của nhau
        # print("Increase sim between: ", i + 1, j)
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

      #Insert to database
      u_sim.insert(0, 'user_1', i, allow_duplicates=False)
      conn = MySQLdb.connect(host="66.42.59.144", user="lucifer", passwd="12344321", db="moviedb")
      object_list = u_sim.to_records(index=False).tolist()
      fields = ("id_user_1", "id_user_2", "similarity")
      upsert(conn, "jaccard_similarity", fields, object_list)
      # print('u_sim: \n',u_sim)
      # print("user_1 " + str(i) +" upserts successfully")
      bar()

# upsert
# purpose: Insert records if not exists, Update records on the dupplicate key
# params:
#     conn is MySQLdb.connect
#     table is the table name need to be inserted data, datatype is str
#     fields are the column names of table, datatype is tuple
#     object_list is the list of dataframe values, datatype is list
def upsert(conn, table, fields, object_list):
    cursor = conn.cursor()
    table = "`"+table+"`"
    fields = ["`"+field+"`" for field in tuple(fields)]
    placeholders = ["%s" for field in fields]
    assignments = ["{x} = VALUES({x})".format(
        x=x
    ) for x in fields]

    query_string = """INSERT INTO
    {table}
    ({fields})
    VALUES
    ({placeholders})
    ON DUPLICATE KEY UPDATE {assignments}""".format(
                                                    table=table,
                                                    fields=", ".join(fields),
                                                    placeholders=", ".join(placeholders),
                                                    assignments=", ".join(assignments)
                                                    )
    
    cursor.executemany(query_string, object_list)
    conn.commit()
    # print(table + ' upserts successfully')

