import pickle
from models import *
import streamlit as st
import torch,torchvision
import os
import cv2
import torch
import numpy as np
from datetime import datetime,timedelta
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
torch.set_warn_always(False)
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
# Function to find all files N days after the date the file was stored
def find_files(usr_name, days_list):
    # List to store the matching files
    matching_files = []
    
    for filename in os.listdir():
        # Extract date from filename
        file_date_str = filename
        #st.write(file_date_str.split("."))
        format = file_date_str.split(".")
        if len(format)>1:
            if format[1] == "png":
                if file_date_str.split("-")[0] == usr_name:
        
                    file_date = datetime.strptime(file_date_str.split("-")[1].replace(".png",""), '%Y_%m_%d')
                    # Check if file matches any of the days in the list
                    for days in days_list:
                        target_date = file_date + timedelta(days=days)
                        target_date_str = target_date.strftime('%Y_%m_%d')
                        if os.path.exists(target_date_str+".png"):
                            matching_files.append([target_date_str+".png",days])

    return matching_files

def clearUp(dataset_path):
    for datas in dataset_path:
        if datas[1] == 15:
            os.remove(datas[0])
    return True
def logout():
    st.session_state.username_flag=False
    st.session_state.user = ""
    st.rerun()
if "install_font" not in st.session_state:
    st.session_state.install_font = False
if "username_flag" not in st.session_state:
    st.session_state.username_flag = False

if "input_size" not in st.session_state:
    st.session_state.input_size = [1024, 1024]
if "net" not in st.session_state:
    st.session_state.net = ISNetDIS()
    st.session_state.net.load_state_dict(torch.load("isnet.pth", map_location="cpu"))
if st.session_state.username_flag:
    dataset_path=f"{st.session_state.user}_wr"  #Your dataset path
    model_path="isnet.pth"  # the model path
    result_path=f"{st.session_state.user}_res"  #The folder path that you want to save the results
    st.title(f"你好！{st.session_state.user}")
    times = [0,2,4,7,15]
    review_list = find_files(st.session_state.user, times)
    
    os.system("ls")
    st.write(review_list)
    if len(review_list) == 0:
        delete_page = st.Page("wrong_questions.py", title="错题分析", icon=":material/notification_important:")
        create_page = st.Page(logout, title="登出", icon=":material/logout:")
        learning = st.Page("learning_curve.py", title="记忆曲线", icon="📉")
        calendar = st.Page("calender.py", title="打卡记录", icon = ":material/dashboard:")
        pg = st.navigation({"面板":[calendar],"打卡":[delete_page],"账号":[create_page]})
    else:
        done = clearUp(review_list)
        if done:
            pg=st.navigation([st.Page("review.py", title="复习")])
else:
    pg = st.navigation([st.Page(login)])
pg.run()
