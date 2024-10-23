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
def find_files(directory, days_list):
    # List to store the matching files
    matching_files = []
    if os.path.isdir(directory):
        # Iterate through files in the directory
        for filename in os.listdir(directory):
            # Extract date from filename
            file_date_str = filename
    
            file_date = datetime.strptime(file_date_str.split(".")[0], '%Y/%m/%d')
    
            # Check if file matches any of the days in the list
            for days in days_list:
                target_date = file_date + timedelta(days=days)
                target_date_str = target_date.strftime('%Y/%m/%d')
                if os.path.exists(os.path.join(directory, target_date_str+".png")):
                    matching_files.append([os.path.join(directory,filename),days])

    return matching_files

def clearUp(dataset_path, model_path, result_path):
    
    input_size = [1024, 1024]
    net = ISNetDIS()
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    im_list = [datas[0] for datas in dataset_path]
    with torch.no_grad():
        for i, im_path in enumerate(im_list):
            im = cv2.imread(im_path)
            if len(im.shape) < 3:
                im = np.stack([im] * 3, axis=-1)  # Convert grayscale to RGB
            im_shp = im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
            image = torch.divide(im_tensor, 255.0)
            image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
            if torch.cuda.is_available():
                image = image.cuda()
            result = net(image)
            result = torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result - mi) / (ma - mi)
            result = result.unsqueeze(0) if result.dim() == 2 else result  # Ensure result has 3 channels
            result = result.repeat(3, 1, 1) if result.shape[0] == 1 else result
            result = 1 - result  # Invert the mask here

            if torch.cuda.is_available():
                result = result.cuda()  # Move result to GPU if available

            im_name = im_path.split('\\')[-1].split('.')[0]

            # Resize the image to match result dimensions
            image_resized = F.upsample(image, size=result.shape[1:], mode='bilinear')

            # Ensure both tensors are 3D
            image_resized = image_resized.squeeze(0) if image_resized.dim() == 4 else image_resized
            result = result.squeeze(0) if result.dim() == 4 else result

            # Apply threshold to result to ensure only pure black or white pixels
            threshold = 0.80  # Adjust as needed
            result[result < threshold] = 0
            result[result >= threshold] = 1

            distance = np.sqrt(np.sum((im - [255, 255, 255]) ** 2, axis=-1))

            # Create a mask where the distance is less than the threshold
            mask = distance < 200

            # Convert mask to uint8
            mask = mask.astype(np.uint8) * 255

            mask = np.stack([mask] * 3, axis=-1)


            result = (result.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
            # result=result.cpu().numpy().astype(np.uint8)
            # io.imsave(result_path + im_name + "_foreground.png", foreground)
            wite = np.ones_like(im) * 255
            cropped = np.where(result == 0, wite, mask)
            cv2.imwrite(result_path + im_name + "_background.png", cropped)
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
    st.session_state.net.load_state_dict(torch.load(model_path, map_location="cpu"))
if st.session_state.username_flag:
    dataset_path=f"{st.session_state.user}_wr"  #Your dataset path
    model_path="isnet.pth"  # the model path
    result_path=f"{st.session_state.user}_res"  #The folder path that you want to save the results
    st.title(f"你好！{st.session_state.user}")
    times = [0,2,4,7,15]
    review_list = find_files(dataset_path, times)
    os.system("ls")
    if len(review_list) == 0:
        delete_page = st.Page("wrong_questions.py", title="错题分析", icon=":material/notification_important:")
        create_page = st.Page(logout, title="登出", icon=":material/logout:")
        learning = st.Page("learning_curve.py", title="记忆曲线", icon="📉")
        calendar = st.Page("calender.py", title="打卡记录", icon = ":material/dashboard:")
        pg = st.navigation({"面板":[calendar],"打卡":[delete_page],"账号":[create_page]})
    else:
        #done = clearUp(review_list, model_path, result_path)
        if done:
            pg=st.navigation([st.Page("review.py", title="复习")])
else:
    pg = st.navigation([st.Page(login)])
pg.run()
