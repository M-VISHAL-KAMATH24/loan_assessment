# -*- coding: utf-8 -*-
"""loan_approval.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RG6M0TgqEoIh6tPXdEil0gTjDeNvuf1w
"""

import pandas as pd

data = pd.read_csv('/content/loan_approval_dataset.csv')
data

data.drop(columns=['loan_id'],inplace=True)

data.columns

data.columns=data.columns.str.strip()

data.columns

data['Assets']=data.residential_assets_value+data.commercial_assets_value+data.luxury_assets_value+data.bank_asset_value



data.drop(columns=['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value'],inplace=True)

data



data.isnull().sum()
#to check wether there is an missing value in the  columns

data.dropna(inplace=True)
#if there is any missing value eliminate those values

data.education.unique()

def clean_data(st):
  st=st.strip()
  return st

clean_data('Graduate')

data.education=data.education.apply(clean_data)

data.education.unique()

data['education']=data['education'].replace(['Graduate', 'Not Graduate'],[1,0])

data.self_employed.unique()
data.self_employed=data.self_employed.apply(clean_data)
data['self_employed']=data['self_employed'].replace(['Yes','No'],[1,0])

data.self_employed.unique()
data['self_employed']=data['self_employed'].replace(['Yes','No'],[1,0])







data

data.loan_status=data.loan_status.apply(clean_data)
data.loan_status.unique()
data['loan_status']=data['loan_status'].replace(['Approved','Rejected'],[1,0])



data['self_employed'] = data['self_employed'].replace(['Yes', 'No'], [1, 0])
data

data



from sklearn.model_selection import train_test_split
input_data=data.drop(columns=['loan_status'])
output_data=data['loan_status']

input_data

output_data

x_train,x_test,y_train,y_test=train_test_split(input_data,output_data,test_size=0.2)

x_test.shape,x_train.shape,y_train.shape,y_test.shape
#854 rows of 8 columns in the input data is for test and 3418 is for the train
#3415 rows of the 1 column of the output data is for the train and 854 for the test
#test size is like 0.2 means 20 percent of the data is fortrhe testing

from sklearn.preprocessing import StandardScaler
#scaling fior speeding up the training and model performance

scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train_scaled,y_train)



model.score(x_test_scaled,y_test)



pred_data=pd.DataFrame([['2'	,'1',	'0'	,'9600000',	'29900000',	'12',	'778','50700000']],columns=['no_of_dependents',	'education'	,'self_employed',	'income_annum',	'loan_amount',	'loan_term',	'cibil_score',	'Assets'])

pred_data=scaler.transform(pred_data)

data

model.predict(pred_data)

import pickle as pk
pk.dump(model,open('model.pkl','wb'))
pk.dump(scaler,open('scaler.pkl','wb'))

from google.colab import files
files.download('model.pkl')
files.download('scaler.pkl')

