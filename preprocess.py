import pandas as pd
import numpy as np


class Preprocessor:
    
    def __init__(self,filename):
        
        self.file = filename

        
    def process(self,train=0.4):
        """
        the main method to process file with ratings
        """

        df = pd.read_csv('ratings.csv').sort_values(by='timestamp')
        
        # as df is sorted by timstamp 
        sep_index = int(train * df.shape[0])
        
        train_df = df[:sep_index]
        test_df = df[sep_index:]
        
        test_df = test_df.loc[test_df.userId.isin(train_df.userId)]
        test_df = test_df.loc[test_df.movieId.isin(train_df.movieId)]
        
        users = set(train_df.userId.values)
        movies = set(train_df.movieId.values)

        movie_index = pd.DataFrame(movies,columns=['movieId']).reset_index()
        movie_coder = {value: key for (key,value) in movie_index.values}
        movie_decoder = {key: value for (key,value) in movie_index.values}

        user_index = pd.DataFrame(users,columns=['userId']).reset_index()
        user_coder = {value: key for (key,value) in user_index.values}
        user_decoder = {key: value for (key,value) in user_index.values}
        
        # transform to appropriate form for als and my method

        train_df.userId = train_df.userId.apply(lambda x: user_coder[x])
        train_df.movieId = train_df.movieId.apply(lambda x: movie_coder[x])

        test_df.loc[test_df.movieId.isin(movies)]
        test_df.userId = test_df.userId.apply(lambda x: user_coder[x])
        test_df.movieId = test_df.movieId.apply(lambda x: movie_coder[x])
        
        print('saving files')
        train_df.to_csv('train_df.csv',index=None)
        test_df.to_csv('test_df.csv',index=None)
        movie_index.to_csv('movieId_index.csv',index=None)
        user_index.to_csv('userId_index.csv',index=None)
        
        return 0
    
    
    
