import streamlit as st
st.write('chao ban')
st.title('ti le tot nghiep')
name = st.text_input('bai cuoi ki')
st.pyplot()
import pandas as pd
df = pd.read_csv('graduation_rate.csv')({
    'firt column' : [1,2,3,4],
    'second column': [10,20,30,40]
})
st.write('graduation_rate.csv',df)
