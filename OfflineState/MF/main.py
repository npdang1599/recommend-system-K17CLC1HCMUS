from recommend_system import RecSys
import pandas as pd


def get_MF_data(RecSys):

    # user_factors
    df1 = pd.DataFrame(RecSys.user_factors)
    df1.to_csv("./result/user_factors.csv", sep=',')

    # user_biases
    df1 = pd.DataFrame(RecSys.user_biases)
    df1.to_csv("./result/user_biases.csv", sep=',')

    # item_factors
    df1 = pd.DataFrame(RecSys.item_factors)
    df1.to_csv("./result/item_factors.csv", sep=',')

    # item_biases
    df1 = pd.DataFrame(RecSys.item_biases)
    df1.to_csv("./result/item_biases.csv", sep=',')

    x = RecSys.ratings_global_mean
    f = open("./result/globalmean.txt","w")
    f.write(f"{x}")

def main():
    gr = RecSys()
    gr.sgd_factorize()
    get_MF_data(gr)

    
if __name__ == "__main__":
    main()