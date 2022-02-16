
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class Data_Prep(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.cat_features = ['Gender', 'Dependent_count', 'Education_Level', 'Marital_Status',
       'Income_Category', 'Card_Category', 'Total_Relationship_Count',
       'Months_Inactive_12_mon', 'Contacts_Count_12_mon']
        self.num_features = ['Customer_Age', 'Months_on_book', 'Credit_Limit', 'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
    
    def fit(self, X, y=None):
        X = X.copy()
        
        self.OneHotEncoder = OneHotEncoder(drop='if_binary', sparse=False)
        
        self.OneHotEncoder.fit(X[self.cat_features])
        
        self.feature_names = self.num_features + ['Income_Credit_ratio'] + self.OneHotEncoder.get_feature_names_out().tolist() 
        
        return self 
    
    def transform(self, X, y=None):
        X = X.copy()
        X = X.drop('CLIENTNUM',axis=1)
        
        Income_Credit_ratio_feature = X.Income_Category.map({'$120K +': 140, '$40K - $60K': 50, '$60K - $80K': 70, 
                                                                    '$80K - $120K': 100,'Less than $40K': 20, 
                                                                    'Unknown': 0}).astype('int') / X.Credit_Limit
        
        cat_features = self.OneHotEncoder.transform(X[self.cat_features])
        
        num_features = X[self.num_features]

        X = np.concatenate((num_features,Income_Credit_ratio_feature.to_numpy()[:,np.newaxis],cat_features), axis=1) 
        
        return X
