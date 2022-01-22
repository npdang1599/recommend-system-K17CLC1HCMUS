import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import fetch_data

pd.set_option('display.max_colwidth', 0)
pd.set_option('display.max_columns', 0)

# documents_df=pd.DataFrame(documents,columns=['documents'])
def description_prepare(cur):
    # documents_df =  pd.read_csv("/content/drive/MyDrive/K17-FinalProject-Recommend Algorithm/Data/res/movies_description.csv", names=['id','documents'])

    documents_df = fetch_data.movie_decription(cur)

    documents_df = documents_df.dropna()
    documents_df = documents_df.reset_index(drop=True)

    # removing special characters and stop words from the text
    stop_words_l=stopwords.words('english')
    documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )
    return documents_df

def most_similar(doc_id,similarity_matrix, documents_df):
    print (f'Document: {documents_df.iloc[doc_id]["documents"]}')
    print ('\n')
    print (f'Similar Documents using Cosine Similarity:')

    similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]

    sim_info = []
    for ix in similar_ix:
        if ix==doc_id:
            continue
        sim_info.append([doc_id, documents_df.iloc[ix]["id"], similarity_matrix[doc_id][ix]])
        sim_df = pd.DataFrame(sim_info, columns=['id_1','id_2',"similarity"])
    return sim_df

def description_similarities(documents_df):
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    document_embeddings = sbert_model.encode(documents_df['documents_cleaned'])

    pairwise_similarities=cosine_similarity(document_embeddings)
    # print(pairwise_similarities)
    return pairwise_similarities

# pairwise_differences=euclidean_distances(document_embeddings)
# most_similar(0,pairwise_similarities,'Cosine Similarity')
# most_similar(0,pairwise_differences,'Euclidean Distance')

 