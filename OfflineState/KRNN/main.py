import KRNN_jaccard_sim
import fetch_data
import MySQLdb

def main():
    conn = MySQLdb.connect(host="66.42.59.144", user="lucifer", passwd="12344321", db="moviedb")
    cur = conn.cursor()

    click_data = fetch_data.rating_click_df(cur)
    click_data.dropna()
    cur.close()

    first_user = click_data['id_user'].min()
    last_user = click_data['id_user'].max()

    sim_df = KRNN_jaccard_sim.get_list_sim(click_data, first_user, last_user)
    # path = "./jaccard_sim/"
    KRNN_jaccard_sim.recal_sim(sim_df,1,10,first_user, last_user)

if __name__ == "__main__":
    main()
