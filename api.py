import os
import sys
from flask import Flask
from flask_restful import Resource, Api, reqparse
import pickle
import numpy as np
import pandas as pd

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

sys.path.append(MODEL_PATH)
from Data_Prep import Data_Prep

print("Loading model from: {}".format(MODEL_PATH))
pickle_off = open(MODEL_PATH, 'rb')
model = pickle.load(pickle_off)


app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    def __init__(self):
        self._required_features = ['CLIENTNUM', 'Customer_Age', 'Gender', 
                                   'Dependent_count', 'Education_Level', 'Marital_Status',
                                   'Income_Category', 'Card_Category', 'Months_on_book',
                                   'Total_Relationship_Count', 'Months_Inactive_12_mon',
                                   'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                                   'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                                   'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
        self.cat_features = ['Gender', 'Education_Level', 'Marital_Status',
                             'Income_Category', 'Card_Category']
        self.num_features = ['CLIENTNUM','Customer_Age', 'Months_on_book', 'Credit_Limit', 'Total_Revolving_Bal', 'Dependent_count',
                             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Relationship_Count',
                             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Months_Inactive_12_mon', 'Avg_Utilization_Ratio', 'Contacts_Count_12_mon']

        self.reqparse = reqparse.RequestParser()
        for feature in self.num_features:
            self.reqparse.add_argument(
                feature, type = float, required = True, location = 'json',
                help = 'No {} provided'.format(feature))
        for feature in self.cat_features:
            self.reqparse.add_argument(
                feature, type = str, required = True, location = 'json',
                help = 'No {} provided'.format(feature))
        super(Prediction, self).__init__()

    def post(self):
        args = self.reqparse.parse_args()
        X = {f: [args[f]] for f in self._required_features}
        X = pd.DataFrame(data=X)
        y_pred = model.predict(X)
        return {'prediction': y_pred.tolist()[0]}

api.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')