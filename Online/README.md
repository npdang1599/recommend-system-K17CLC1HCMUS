# recommend-system-K17CLC1HCMUS
-----------------------------------FOUR RECOMMEND API:----------------------------------
1. URL: /individual/state1?id_user=...&n_movie=... 
return recommend list of movies for an input user base on implicit feedback click (state 1) what api does: Detect user is new user or not:
    - if is new user: base on ids of the movie user has just chosen from the survey and apply content-based recommend algorithm based on movies' genre to find relevant movies
    - if not new user: apply K-RNN algorithm based on implicit feedback click

2. URL: /individual/state2?id_user=...&n_movie=...&id_movie=...&filter=...(description, genre, default)
return recommend list of movies for an input user base on implicit feedback time watch (state 2) what api does: Detect user is new user or not:
    - if is new user: base on id of the movie user has just watched and apply item-item content-based recommend system to find relevant movies
    - if not new user: Matrix factorization Col-filtering + bert embedding + KNN

3. URL: /group/state1?id_user=...&id_user=...&...&n_movie=...
return recommend list of movies for a list of input user base on implicit feedback click (state 1) what api does: merge state 1 individual recommend list of each user based on their similarity (new user has 0 simmilarity)

4. URL: /group/state2?id_user=...&id_user=...&...&n_movie=...&id_movie=...
return recommend list from groupMF algorithm, if group is all new user, list movie comes from item-item algorithm 

-----------------------------------RECOMMEND ALGORITHMS:----------------------------------

Algorithms use in recommend system:
- new_user: K-nearest neighbor
- individual:
	+ state 1: KRNN base on user's click data
	+ state 2: matrix fractorization base on watching time data
		filter by: genres, description

- group:
	+ state 1: join each member's recommend result from individual state 1
	+ state 2: matrix fractorization for group: before fractorizarion method base on users' watching time data
		filter by: genres, description

-----------------------------------FILES OVERVIEW:----------------------------------

- Latent factor model: (for both individual and group)
	+ GroupMF_group.py: Group class
	+ GroupMF_recommend_engine.py: RecSys class: online state algorithm implement of GroupMF
	+ GroupMF_config.py: Config class: read config file for GroupMF algorithm
- K-RNN:
	+ KRNN_recommend_engine.py: implement online state of K-RNN colaborative algorithm 
- K-NN:
	+ cold_start_KNN_genre.py: perform KNN algorithm on genre data base on cosine similarity
	+ cold_start_KNN_time_watched.py: perform KNN algorithm on time_watched data base on cosine similarity
- Addition:
	+ utils:py: ultility functions 
- main file:
	+ api.py: create recommend api
