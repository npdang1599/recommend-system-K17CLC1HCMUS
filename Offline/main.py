from recommend_system import RecSys
import pandas as pd
import MySQLdb
from alive_progress import alive_bar
import KRNN_jaccard_sim
import fetch_data
import flask
from flask import render_template
from flask_mysqldb import MySQL
from datetime import datetime

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    main()
    return """<h1>Finished</h1>"""

def get_MF_data(RecSys, conn):
    # user_factors
    df = pd.DataFrame(RecSys.user_factors)
    df.index = df.index + 1 #convert index(0,1,2,..,n) to index(1,2,3,...,n)
    object_list = df.to_records(index=True).tolist()
    fields = ("id_user", "factor_1", "factor_2", "factor_3", "factor_4", "factor_5"
            , "factor_6", "factor_7", "factor_8", "factor_9", "factor_10"
            , "factor_11", "factor_12", "factor_13", "factor_14", "factor_15")
    upsert(conn, "user_factors", fields, object_list)        
    yield

    # user_biases
    df = pd.DataFrame(RecSys.user_biases)
    df.index = df.index + 1 #convert index(0,1,2,..,n) to index(1,2,3,...,n)
    object_list = df.to_records(index=True).tolist()
    fields = ("id_user", "user_bias")
    upsert(conn, "user_biases", fields, object_list)
    yield

    # item_factors
    df = pd.DataFrame(RecSys.item_factors)
    df.index = df.index + 1 #convert index(0,1,2,..,n) to index(1,2,3,...,n)
    object_list = df.to_records(index=True).tolist()
    fields = ("id_movie", "factor_1", "factor_2", "factor_3", "factor_4", "factor_5"
            , "factor_6", "factor_7", "factor_8", "factor_9", "factor_10"
            , "factor_11", "factor_12", "factor_13", "factor_14", "factor_15")
    upsert(conn, "movie_factors", fields, object_list)
    yield

    # item_biases
    df = pd.DataFrame(RecSys.item_biases)
    df.index = df.index + 1 #convert index(0,1,2,..,n) to index(1,2,3,...,n)
    object_list = df.to_records(index=True).tolist()
    fields = ("id_movie", "movie_bias")
    upsert(conn, "movie_biases", fields, object_list)
    yield

    #global_mean
    df = pd.DataFrame({'rating': RecSys.ratings_global_mean}, index=[1])
    object_list = df.to_records(index=True).tolist()
    fields = ("id", "rating")
    upsert(conn, "global_mean_ratings", fields, object_list)
    yield


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
    print(table + ' upserts successfully')


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


def get_list_of_old_users(conn):
    cur = conn.cursor()
    cur.execute("""SELECT id_user
                    FROM moviedb.interactive
                    WHERE time_watched > 2
                    GROUP BY id_user
                    HAVING COUNT(id_user) > 20;
                    """)
    res = cur.fetchall()
    result = [i[0] for i in res]
    return result

def update_old_user_flag(conn, list_old_users):
  cur = conn.cursor()
  cur.execute("""UPDATE moviedb.user
                SET is_old_user = 1
                WHERE is_old_user <> 1  AND id in %s;""",(list_old_users,))

  conn.commit()
  print("Update old user flag successfully!")

def update_refresh_date(conn, refresh_date):
    cur = conn.cursor()
    table = 'refresh_model_date'
    fields = ('id', 'refresh_date')
    object_list = refresh_date.to_records(index=True).tolist()
    # print('object_list: ', object_list)
    upsert(conn, table, fields, object_list)
    conn.commit()
    print("Update refresh date successfully") 

def main():
    # get refresh model date
    current_datetime = datetime.now()
    current_datetime_df = pd.DataFrame(data={'time':str(current_datetime)}, index = [1])

    conn = MySQLdb.connect(host="66.42.59.144", user="lucifer", passwd="12344321", db="moviedb")
    list_old_users = get_list_of_old_users(conn)
    # kRNN ------------------------------------
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

    # MF ---------------------------------------
    gr = RecSys()
    gr.sgd_factorize()
    print("Upserting user factors, user biases, item factors, item biases, global mean: ...")
    with alive_bar(5) as bar:
        for i in get_MF_data(gr, conn):
            bar()

    update_old_user_flag(conn, list_old_users)
    update_refresh_date(conn, current_datetime_df)
    conn.close()
    
app.run()