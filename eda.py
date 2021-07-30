import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def EDA():
            train_df=pd.read_csv("data_banknote_authenticationEDA.csv")                  
            train_df = train_df.sample(frac = 1)               
            sns.pairplot(train_df)
            
            plt.show()
            train_df.boxplot()
            plt.show()
            train_df.hist()
            plt.show()

            from pandas.plotting import scatter_matrix
            scatter_matrix(train_df, alpha=0.2, figsize=(6, 6), diagonal='kde')
            plt.show()



            



EDA()
