import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.write('Chào bạn!')
st.title('Tỷ lệ tốt nghiệp')

# Text input
name = st.text_input('Bài cuối kỳ')

# Load and display DataFrame
df = pd.DataFrame({
    'First column': [1, 2, 3, 4],
    'Second column': [10, 20, 30, 40]
})

st.write('Graduation rate DataFrame:')
st.write(df)

# Example plot
fig, ax = plt.subplots()
ax.plot(df['First column'], df['Second column'])
st.pyplot(fig)

