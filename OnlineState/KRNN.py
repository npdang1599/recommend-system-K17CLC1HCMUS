import pandas as pd
import utils


#Hàm trả về danh sách item tư vấn cho người dùng
#Input: 
#user - id người dùng đang xét, N - số lượng item, training_df - file training, sim_path - đường dẫn thư mục độ tương đồng
#Output:
#Danh sách N item tư vấn cho người dùng
def recommend_sys(user, N, training_df, user_df):
    item_col=[]
    rating_col=[]

    # start = time.time()
    #Lấy ra độ tương đồng của người dùng 
    #user_df = pd.read_excel(r"{0}/{1}.xlsx".format(sim_path, user))
    ratings = utils.convert_data_to_array(training_df)
    #các item của người dùng có đánh giá tính cực
    #user_item = training_df.id_movie[training_df.id_user == user][training_df.rating == 1].to_list()

    #danh sách các item cần dự đoán
    #predict_list = [value for value in range(1,3953) if value not in user_item]
    predict_list = utils.find_candidate_items(ratings, [user])
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
    return df, df['Item'].to_list()
