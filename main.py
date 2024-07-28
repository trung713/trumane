import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.write('Chào bạn!')
st.title('Tỷ lệ tốt nghiệp')

# Nhập văn bản
name = st.text_input('Bài cuối kỳ')

# Tải DataFrame từ CSV
df = pd.read_csv('graduation_rate.csv')

st.write('Dữ liệu tỷ lệ tốt nghiệp:')
st.write(df)

# Vẽ biểu đồ mẫu
fig, ax = plt.subplots()
ax.plot(df['Cột thứ nhất'], df['Cột thứ hai'])
st.pyplot(fig)
