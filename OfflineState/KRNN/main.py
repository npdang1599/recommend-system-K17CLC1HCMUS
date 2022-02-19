import KRNN_jaccard_sim
import fetch_data
import MySQLdb

def main():
    conn = MySQLdb.connect(host="66.42.59.144", user="lucifer", passwd="12344321", db="moviedb")
    cur = conn.cursor()

    click_data = fetch_data.rating_click_df(cur)
    click_data.dropna()
    cur.close()

    user_lst = click_data.id_user.drop_duplicates().to_list()
    # print('user_lst: ',user_lst)

    # print("Rating_click_df:\n", click_data)
    sim_df = KRNN_jaccard_sim.get_list_sim(click_data,user_lst)
    # print("sim_df:\n", sim_df)
    KRNN_jaccard_sim.recal_sim(sim_df,1,2,user_lst)

if __name__ == "__main__":
    main()
