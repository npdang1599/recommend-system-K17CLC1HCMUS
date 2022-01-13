import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def get_genre(mysql):
    cur = mysql.connection.cursor()

    cur.execute("""SELECT mList.id_movie, group_concat( li.name)
                    FROM moviedb.movie_list mList
                    JOIN moviedb.list li ON mList.id_list = li.id
                    where li.type = 0
                    group by mList.id_movie""")
    res = cur.fetchall()
    mysql.connection.commit()
    movies = pd.DataFrame(res, columns=['movieId','genres'])
    cur.close()
    
    movies['genres'] = movies['genres'].apply(lambda x: x.split(","))
    
    genres_counts = Counter(g for genres in movies['genres'] for g in genres)
    
    genres = list(genres_counts.keys())

    for g in genres:
        movies[g] = movies['genres'].transform(lambda x: int(g in x))
    
    cosine_sim = cosine_similarity(movies[genres], movies[genres])
    print(f"Dimensions of our movie features cosine similarity matrix: {cosine_sim.shape}")
    
    return movies[genres]

def check_new_user(movie_ids):
    return len(movie_ids) < 20

def get_movie_ids_from_db(mysql, id):
    cur = mysql.connection.cursor()
    cur.execute("""SELECT id_user ,GROUP_CONCAT(id_movie) FROM moviedb.interactive WHERE id_user = %s AND is_clicked <> 0""",(id,))
    res = cur.fetchall()
    mysql.connection.commit()
    cur.close()

    res = res[0][1]
    ids = res.split(',')
    ids = [int(s) for s in ids]

    return ids

def cosine_sim(features):
    return cosine_similarity(features)
    
def get_content_based_recommendations(idx,cosine_sim, n_recommendations=10):
    sim_scores = []
    # print('len idx: ',len(idx))
    # print('idx: ',idx)


    for i in range(len(idx)):
        tmp = list(enumerate(cosine_sim[idx[i]]))
        
        sim_scores.extend(tmp)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    print("sim_scores: ", sim_scores)
    similar_movies = [i[0] for i in sim_scores]
    
    return similar_movies

def get_recommend_list(list_item_ids,n_recommendations, mysql):
    cosine_sim_mtrx = cosine_sim(get_genre(mysql))
    res = get_content_based_recommendations(list_item_ids, cosine_sim_mtrx, n_recommendations)
    return res

    
