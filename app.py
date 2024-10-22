import pickle

import streamlit as st
import os
from datetime import datetime,timedelta
if "role" not in st.session_state:
    st.session_state.role = None
st.set_page_config(page_title="打卡APP", page_icon=":material/edit:")
def login():
    st.header("登录")
    text = st.empty()
    def checkN(x):
        username = text.text_input("请输入你的姓名:", value="", key=x)
        if username != "":
            
            if not username in student_names:
                text.write("<font style='red'>无效姓名</font>",key=x)
                checkN(x+1)
            else:
                st.session_state.username_flag=True
                st.session_state.user = username
                return
    
    student_names = ["李天恩"]
    checkN(0)
    if st.session_state.username_flag:
        st.rerun()
def logout():
    st.session_state.username_flag=False
    st.session_state.user = ""
    st.rerun()
if "install_font" not in st.session_state:
    st.session_state.install_font = False
if "username_flag" not in st.session_state:
    st.session_state.username_flag = False
if st.session_state.username_flag:
    st.title(f"你好！{st.session_state.user}")
    times = [1,2,4,7,15]
    review_list = []
    if os.path.exists(f"{st.session_state.user}_reviews.pkl"):

        reviews = pickle.load(open(f"{st.session_state.user}_reviews.pkl","rb"))
        for rs in reviews:
            for time in times:
                date_dt = rs[0]+timedelta(days=time)
                if date_dt.strftime("%Y/%m/%d") == datetime.now().strftime("%Y/%m/%d"):
                    review_list.append(rs[1])
    if len(review_list) == 0:
        delete_page = st.Page("wrong_questions.py", title="错题分析", icon=":material/notification_important:")
        create_page = st.Page(logout, title="登出", icon=":material/logout:")
        learning = st.Page("learning_curve.py", title="记忆曲线", icon="📉")
        calendar = st.Page("calender.py", title="打卡记录", icon = ":material/dashboard:")
        pg = st.navigation({"面板":[calendar],"打卡":[delete_page],"账号":[create_page]})
    else:
        pg=st.navigation([])
else:
    pg = st.navigation([st.Page(login)])
pg.run()
