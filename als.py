from implicit.als import AlternatingLeastSquares as als

from scipy.sparse import csr_matrix
import scipy.sparse as sp
from loguru import logger
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from validation import Validator


class ALS_helper:
    
    def __init__(self,factors=10,train_df_path='train_df.csv',test_df_path='test_df.csv'):
        
        self.model = als(factors=factors)
        self.train_df = pd.read_csv(train_df_path)
        self.test_df = pd.read_csv(test_df_path)
        self.ratings_matrix = self.__get_matrix()
        self.user_items = self.ratings_matrix.T.tocsr()
        self.validator = Validator(test_df_path)
        
        os.environ["OPENBLAS_NUM_THREADS"] = '1'
        logger.info('The heplper is initialized succesfully')
        
        
    def train(self):
        logger.info('Model is training now')
        self.model.fit(self.ratings_matrix)
        logger.info('Model is trained')
        
        
    def __recommend(self):
        recom_list = self.model.recommend_all(self.user_items,filter_already_liked_items=True)
        return recom_list
    
    def validate(self):
        
        users = np.unique(self.test_df.userId.values)
        recom_list = self.__recommend()
        
        ndc1_als = []
        ndc10_als = []

        for user in tqdm(users):
            n1,n10 = self.validator.valid(user,recom_list[user])

            ndc1_als.append(n1)
            ndc10_als.append(n10)
        
        return np.mean(ndc1_als),np.mean(ndc10_als)
        
    def __get_matrix(self):
        
        self.train_df['userId'] = self.train_df['userId'].astype('category')
        self.train_df['movieId'] = self.train_df['movieId'].astype('category')

        ratings_matrix = sp.coo_matrix(
            (self.train_df['rating'].astype(np.float32) ,
                (
                    self.train_df['movieId'].cat.codes.copy(),
                    self.train_df['userId'].cat.codes.copy()
                )
            )
        )
        
        ratings_matrix = ratings_matrix.tocsr()
        
        return ratings_matrix
    

mdl = ALS_helper()
mdl.train()
a,b = mdl.validate()
logger.info('ndcg@1 = {}, ndcg@10 = {}'.format(a,b))
