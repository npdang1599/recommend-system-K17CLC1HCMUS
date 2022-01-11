import pandas as pd
import numpy as np
import flask
from flask_mysqldb import MySQL
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

def cosine_sim(features):
    return cosine_similarity(features, features)
    
def get_content_based_recommendations(idx,cosine_sim, n_recommendations=10):
    sim_scores = []
    for i in range(len(idx)):

        tmp = list(enumerate(cosine_sim[idx[i]]))
        for j in range(len(tmp)-1):
            if tmp[j-1][0] in idx:
                #print()
                tmp.remove(tmp[j])
        #print(tmp[0][0]==1)
        sim_scores.extend(tmp)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    print(type(sim_scores))
    similar_movies = [i[0] for i in sim_scores]
    
    return similar_movies