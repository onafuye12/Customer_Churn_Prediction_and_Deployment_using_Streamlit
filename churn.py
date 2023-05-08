
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

model = pickle.load(open('RF_class_model.pkl','rb'))
scaler = pickle.load(open('scal_class.pkl', 'rb'))
encoder = pickle.load(open ('enc_class.pkl', "rb"))

df = st.file_uploader('upload a csv',type='csv')


#if(not df):
   # st.info('The prediction will begin, once you upload your data set')
   # st.stop()
    
if df is not None:
    #read csv file into a dataframe
    df = pd.read_csv(df)
else:
    st.stop()

def prep(df):
    #df.reset_index(drop=True, inplace=True)
    df['TotalCharges'] = df['TotalCharges'].replace(' ',np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df.drop_duplicates(inplace=True,keep='first',ignore_index=True)
    df = df.dropna().reset_index(drop=True)
    cust = df['customerID']
    df.drop(['customerID'],axis=1,inplace=True)
    #df = df.drop(['Churn'],axis=1)
    #y_test = df['Churn']
    cat1 = []
    for i in df.columns:
        if df[i].dtype == 'O':
            cat1.append(i)
    enc_data =pd.DataFrame(encoder.transform(df[cat1]).toarray())
    #enc_data.columns = encoder.get_feature_names_out()
    enc_data.columns = encoder.get_feature_names(cat1)
    df = df.join(enc_data)

    df.drop(cat1,axis=1,inplace=True)
    col1 = df.columns
    df = scaler.transform(df)
    df = pd.DataFrame(df,columns=col1)
    return cust, df

cust , c_data = prep(df)
pred = model.predict(c_data)
results = pd.DataFrame({'Cust_ID':cust,'Churn_pred':pred})
targ_cust = results[results['Churn_pred'] == 'Yes'].reset_index(drop=True)['Cust_ID']

c1,c2 = st.columns(2)

with c1:
    if st.button('Prediction'):
        st.dataframe(results)
        csv1 = results.to_csv(index=False)
        st.download_button('Download Predictions', csv1,file_name='predictions.csv')
        
with c2:
    if st.button('Churn Customers'):
        st.dataframe(targ_cust)
        csv2 = targ_cust.to_csv(index=False)
        st.download_button('Download Target customer list',csv2, file_name = 'churn_cust.csv')
    
