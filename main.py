import streamlit as st

st.title('My project in Streamlit')
st.header('This is a header')
st.subheader('This is a sub header')
st.divider()
st.markdown('#Heading 1')
st.markdown('My lesson for dummy  ')
st.markdown('[Van Lang University](https://)')
st.markdown("""
            1. Machine learning
            2.Deep learing
            """)
st.markdown(r'$\sqrt{2x} $')

st.divider()

st.latex('\sqrt{2x}')

st.divider()
st.write('[Google](http://)')
st.write('$ \sqrt{2x} $')
st.write('1 + 1=',2)
st.divider()
st.image('google.jpg','Funny picture')
# st.audio('kh90.mp4')
# st.divider()
# st.video('kh90.mp4')
# st.divider()

# agree=st.checkbox("I agree!")
# if agree:
#     st.write("Thanks")
# status=st.radio('Your Favorite color: ',['Yellow','Blue'])
# print(status)
# options=st.multiselect('Colors:',['Green','Yellow','Blue'],['Yellow','Blue'])
# print(options)

# st.select_slider('Your color:',['Red','Yellow','Blue'])

# st.divider()
# if st.button('Submit'):
#     st.write("Hello Thái")
# else:
#     st.write("Goodbye")
# name=st.text_input('Your name: ',value='Annna')
# st.write(name)
# st.divider()
# files=st.file_uploader('Nhập file cần upload: ',accept_multiple_files=True)
# for file in files:
#     read_f=file.read()
#     st.write('File name: ',file.name)
# st.divider()
# #with st.form('Infomation of Exercises'):
#  #   col1,col2=st.columns(2)
#   #  f_name=col1.text_input('Name')
#    # f_age=col2.text_input('Age')
#     #sumbmits=st.form_submit_button('Save')
#     #if sumbmits:
#      #   st.write(f"Name: {f_name}, Age: {f_age}")
# st.divider()

# import random
# value1=random.randint(5,40)
# value2=random.randint(10,20)
# if 'key' not in st.session_state:
#     st.session_state['email']=value1
#     st.session_state['password']=value2
# st.write(st.session_state.key)
