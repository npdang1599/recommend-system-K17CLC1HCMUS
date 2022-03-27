import pandas as pd
from collections import Counter
from Online import fetch_data

def get_newly_genre_reference(cur,id_user):
    genre_df = fetch_data.get_list_new_interactive(cur,id_user)
    genre_df['genre'] = genre_df['genre'].apply(lambda x: x.split(","))

    genres_counts = Counter(g for genre in genre_df['genre'] for g in genre)

    return genre_df, pd.DataFrame(list(dict(genres_counts).items()), columns=['genre','count'])

def apply_bias(cur, id_user, predicted_df):
    new_genre_df, new_genre_count_df = get_newly_genre_reference(cur, id_user)
    # print('new_genre_df: ', new_genre_df)
    # print('new_genre_count_df: ', new_genre_count_df)
    genre_df = fetch_data.get_genre(cur)
    predicted_df.columns = ['Item', 'Rating']
    predicted_df = remove_interactive(predicted_df,new_genre_df['Item'].to_list())

    predicted_df = predicted_df.join(genre_df.set_index('Item'), on='Item').fillna('')
    # print('predicted_df: ', predicted_df)
    #print(predicted_df)
    # find_index = predicted_df[predicted_df['Item']==52]
    # print("predicted_df['genre'][52]: ", predicted_df['genre'][find_index])

    for i in range(len(new_genre_count_df['genre'])):
        for j in predicted_df.Item.to_list():
            # print('j: ', j)
            find_index = predicted_df[predicted_df['Item']==j]
            # print("find_index: ", find_index)
            if len(find_index)!=0:
                index = find_index.index.values.astype(int)[0]
                if new_genre_count_df['genre'][i] in predicted_df['genre'][index]:
                    # print("new_genre_count_df['genre'][i]: ", new_genre_count_df['genre'][i])
                    predicted_df['Rating'][index] += new_genre_count_df['count'][i]

    predicted_df = predicted_df.sort_values(by=['Rating'],axis=0,ascending=False, ignore_index=True)
    predicted_df = predicted_df.drop(['genre'], axis=1)
    #print(predicted_df)

    return predicted_df, predicted_df['Item'].to_list()

def remove_interactive(target_df,list_id_remove):
    for i in list_id_remove:
        find_index = target_df[target_df['Item']==i]
        # index = find_index.index.values.astype(int)[0]
        if len(find_index)!=0:
            index = find_index.index.values.astype(int)[0]
            # x=genre_df.drop(index, axis=0)
            target_df=target_df.drop(index, axis=0)
    return target_df





