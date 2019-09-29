import pandas as pd
import numpy as np
from tqdm import tqdm



def func(x):
    a = x.values
    a.sort(0)
    a = a[::-1]
    a = a[:,0]
    return a.tolist()


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

class Validator:
    
    def __init__(self,test_path):
        validation_df = pd.read_csv(test_path)
        self.us_mov = validation_df.groupby(['userId'])[['movieId','rating']].apply(lambda x: func(x)).reset_index()
        self.us_mov.rename({0:'consumed_list'},axis=1,inplace=True)
        
    def valid(self,user,rec_list):
        
        consumed  = self.us_mov.loc[(self.us_mov.userId == user),'consumed_list'].values[0]
       # print(consumed)
        
        rat = [0]*len(rec_list)
        
        for i,el in enumerate(rec_list):
            if el in consumed:
                rat[i] = 1
            else:
                rat[i] = 0 
    
        return ndcg_at_k(rat,1),ndcg_at_k(rat,10)
        