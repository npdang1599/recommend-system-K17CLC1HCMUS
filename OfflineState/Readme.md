OFFLINE STATE:
- K-RNN: 
    + S1: jaccard_sim.py: Calculate Jaccard Similarity between items
    + S2: KRNN_jaccard_sim.py: Recalculate the similarity when adding an symmetrically improved criteria 
    + S3: main.py: push jaccard similarity to SQL server using upsert fuction
- MF:
    + S1: Apply basic Matrix Factorization algorithm to generate smaller matrixs:  user factor, item factor, user bias and item bias, ... 
    + s2: Push above matrixs to SQL database using upsert fuction
    + Addition: description_sim.py: perform bert embedding and KNN algorithm to find similar items, up the cosine similarity to database
    
