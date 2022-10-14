# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:54:36 2022

@author: somesh
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st



# defining functions of catrgorical and numerical data-types

# below function returns list of all the categorical features present in the dataset
def get_categorical_features(dataset):
    categorical_features = []
    for feature in dataset.columns:
        if dataset[feature].dtypes == 'O':
            categorical_features.append(feature)
    return categorical_features

 # below function returns a list of all the numerical features present in the dataset
def get_numerical_features(dataset):
    numerical_features = []
    for feature in dataset.columns:
        if dataset[feature].dtypes != 'O':
            numerical_features.append(feature)
    return numerical_features

#This function takes a dataset as input and return a list of columns for which contain a null value
def identify_features_na(data):
    features_with_na = []
    for column in data.columns:
        if data[column].isnull().sum() > 1:
            features_with_na.append(column)
    return features_with_na 

def replace_categorical_feature_na(data, categorical_feature_nan):
    dataset = data.copy()
    dataset[categorical_feature_nan] = dataset[categorical_feature_nan].fillna('Missing')
    return dataset

def encode_categorical_data(data, encoders_dict):
    encoded_data = []
    flag = 0
    for column, encoder in encoders_dict.items():
        if flag == 0:
            encoded_data = encoder.transform(data[column].values.reshape(-1,1)).toarray()
            flag = 1
        else:
            encoded_data = np.hstack((encoded_data,encoder.transform(data[column].values.reshape(-1,1)).toarray()))
    return encoded_data



def predict_fn():
    data= df
    
    #handle missing values
    data.drop(['PersonalField84','PropertyField29'],axis=1,inplace=True)
    
    #list of missing value features
    features_categorical = get_categorical_features(data.loc[:,identify_features_na(data)])
    data = replace_categorical_feature_na(data, features_categorical)
    
    #dropping date, quote_number
    quote_number = data['QuoteNumber']
    data.drop(columns = ['QuoteNumber','Original_Quote_Date'], axis = 1, inplace = True)
    
    #onehotencodecategoricaldata
    encoders_dict = pickle.load(open('encoders_dict','rb'))
    encoded_categorical_data = encode_categorical_data(data,encoders_dict)
    data.drop(labels = list(encoders_dict.keys()), axis = 1 , inplace = True)
    data = np.hstack((data.to_numpy(),encoded_categorical_data))
    
    #removing constant,scaling data
    vr = pickle.load(open('constant_features','rb'))
    data = data[:,vr.get_support()]
    scaler = pickle.load(open('feature_scaling','rb'))
    data = scaler.transform(data)
    
    #loading model
    model = pickle.load(open('model','rb'))
    y_pred = np.argmax(model.predict_proba(data), axis = -1)
    
    predictions = pd.DataFrame(data = zip(quote_number,y_pred), columns = ['QuoteNumber','QuoteConversion_Flag'])
    return predictions



def main():
    #giving a title
    st.title('Home insurance Quote conversion')
    
    global df
    
    #getting the input data
    data_file= st.file_uploader("upload csv",type=["csv"])
    if data_file is not None:
        file_details = {"file name":data_file.name, "file type":data_file.type, "file size":data_file.size}
        st.write(file_details)
        
        df=pd.read_csv(data_file).sample(50)
        st.write(df)
    
        quote_converted=''
        
        if st.button('predict'):
            quote_converted= predict_fn()
        
        st.write(quote_converted)
        
        

if __name__ =='__main__':
    main()
