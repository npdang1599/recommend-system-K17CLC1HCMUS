from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
import fetch_data

# get_genre fuction: get genre data of movies
def get_genre(cur):

    genre_df = fetch_data.genre(cur)
    # print(genre_df)
    
    genre_df['genres'] = genre_df['genres'].apply(lambda x: x.split(","))
    genres_counts = Counter(g for genres in genre_df['genres'] for g in genres)
    
    genres = list(genres_counts.keys())

    for g in genres:
        genre_df[g] = genre_df['genres'].transform(lambda x: int(g in x))
    
    # cosine_sim = cosine_similarity(genre_df[genres], genre_df[genres])
    # print(f"Dimensions of our movie features cosine similarity matrix: {cosine_sim.shape}")
    
    return genre_df[genres]

def check_new_user(movie_ids):
    return len(movie_ids) < 20

def get_movie_ids_from_db(cur, id):
    cur.execute("""SELECT id_user ,GROUP_CONCAT(id_movie) FROM moviedb.interactive WHERE id_user = %s AND is_clicked <> 0""",(id,))
    res = cur.fetchall()

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
    # print("sim_scores: ", sim_scores)
    similar_movies = [i[0] for i in sim_scores]
    return similar_movies

def get_recommend_list(list_item_ids,n_recommendations, cur):
    cosine_sim_mtrx = cosine_sim(get_genre(cur))
    res = get_content_based_recommendations(list_item_ids, cosine_sim_mtrx, n_recommendations)
    return res

def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.
    
    Args:
        df: pandas dataframe
    
    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    df.columns = ['userId','movieId','rating']
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

def find_similar_movies(movie_id,k, ratings, metric='cosine', show_distance=False):
    """
    Finds k-nearest neighbours for a given movie id.
    
    Args:
        movie_id: id of the movie of interest
        X: user-item utility matrix
        k: number of similar movies to retrieve
        metric: distance metric for kNN calculations
    
    Returns:
        list of k similar movie ID's
    """
    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

    neighbour_ids = []
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    # print(type(neighbour_ids))
    # print(neighbour_ids)
    return neighbour_ids