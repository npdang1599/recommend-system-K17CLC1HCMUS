from re import X
from unittest import result
from recommend_system import RecSys
import pandas as pd
import MySQLdb
import description_sim

def get_MF_data(RecSys, conn):
    

    # user_factors
    df = pd.DataFrame(RecSys.user_factors)
    df.index = df.index + 1 #convert index(0,1,2,..,n) to index(1,2,3,...,n)
    object_list = df.to_records(index=True).tolist()
    fields = ("id_user", "factor_1", "factor_2", "factor_3", "factor_4", "factor_5"
            , "factor_6", "factor_7", "factor_8", "factor_9", "factor_10"
            , "factor_11", "factor_12", "factor_13", "factor_14", "factor_15")
    upsert(conn, "user_factors", fields, object_list)        

    # user_biases
    df = pd.DataFrame(RecSys.user_biases)
    df.index = df.index + 1 #convert index(0,1,2,..,n) to index(1,2,3,...,n)
    object_list = df.to_records(index=True).tolist()
    fields = ("id_user", "user_bias")
    upsert(conn, "user_biases", fields, object_list)

    # item_factors
    df = pd.DataFrame(RecSys.item_factors)
    df.index = df.index + 1 #convert index(0,1,2,..,n) to index(1,2,3,...,n)
    object_list = df.to_records(index=True).tolist()
    fields = ("id_movie", "factor_1", "factor_2", "factor_3", "factor_4", "factor_5"
            , "factor_6", "factor_7", "factor_8", "factor_9", "factor_10"
            , "factor_11", "factor_12", "factor_13", "factor_14", "factor_15")
    upsert(conn, "movie_factors", fields, object_list)

    # item_biases
    df = pd.DataFrame(RecSys.item_biases)
    df.index = df.index + 1 #convert index(0,1,2,..,n) to index(1,2,3,...,n)
    object_list = df.to_records(index=True).tolist()
    fields = ("id_movie", "movie_bias")
    upsert(conn, "movie_biases", fields, object_list)

    #global_mean
    df = pd.DataFrame({'rating': RecSys.ratings_global_mean}, index=[1])
    object_list = df.to_records(index=True).tolist()
    fields = ("id", "rating")
    upsert(conn, "global_mean_ratings", fields, object_list)
    

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


def description_sim_matrix(conn):
    cur = conn.cursor()
    documents_df = description_sim.description_prepare(cur)
    # print("doc :", documents_df)
    result = description_sim.description_similarities(documents_df)

    return documents_df, result


def main():
    conn = MySQLdb.connect(host="66.42.59.144", user="lucifer", passwd="12344321", db="moviedb")
    # gr = RecSys(conn)
    # gr.sgd_factorize()
    # get_MF_data(gr)
    doc,x = description_sim_matrix(conn)

    for i in doc['id']:
        sim_df = description_sim.most_similar(i, x,doc)
        print(sim_df)
    conn.close()

    
if __name__ == "__main__":
    main()