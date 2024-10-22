import streamlit as st

st.header("Log in")
text = st.empty()
def checkN(x):
    global username_flag
    username = text.text_input("请输入你的姓名:", value="", key=x)
    if username != "":
        
        if not username in student_names:
            st.write("无效姓名")
            checkN(x+1)
        else:
            st.session_state.username_flag=True
            st.session_state.user = username
            return

student_names = ["李天恩"]
checkN(0)
