import streamlit as st
import random
from datetime import datetime
import os
import pickle
import numpy as np
import cv2
import time

st.title("错题总结")
stages = [("知识点总结", "请详细描述这道题的知识点"), ("考点总结", "请详细描述这道题的考点（知识点的拓展)"),
          ("下次如何避免出错", "根据以上提供的答案，描述如和下次同样考点不出错")]
if 'out1' not in st.session_state:
    st.session_state.out1 = None
if 'out2' not in st.session_state:
    st.session_state.out2 = None
if 'out3' not in st.session_state:
    st.session_state.out3 = None
if 'genFlag' not in st.session_state:
    st.session_state.genFlag = False
if "stage" not in st.session_state:
    st.session_state.stage = 1
else:
    if st.session_state.stage==0:
        st.session_state.stage=1
if "kick" not in st.session_state:
    st.session_state.kick = False
if "reruned" not in st.session_state:
    st.session_state.reruned = False
if "thesis" not in st.session_state:
    st.session_state.thesis = 0
if "test" not in st.session_state:
    st.session_state.test = 0
if "prevention" not in st.session_state:
    st.session_state.prevention = 0
if "qtype" not in st.session_state:
    st.session_state.qtype = 0
if "username_flag" not in st.session_state:

    st.write("<font color='red'>请先登录</font>", unsafe_allow_html=True)

else:
    uploaded_file = st.file_uploader("请上传作业图片", type=["png", "jpg"])
    if uploaded_file:

        if "messages" not in st.session_state:
            if os.path.exists(f"{st.session_state.user}.pkl"):
                st.session_state.messages = pickle.load(open(f"{st.session_state.user}.pkl", "rb"))
            else:
                st.session_state.messages = []
        if st.button("删除记录"):
            st.session_state.messages = []
            pickle.dump([], open(f"{st.session_state.user}.pkl", "wb"))
        # Initialize chat history
        with st.chat_message("assistant"):
            response = st.radio("这道题出错的原因是什么？",
                                ["基本功不扎实", "知道知识点但是不知道怎么运用", "完全没有思路", "读题出错"])
            st.progress(0)
            if st.button("提交"):
                if response != "完全没有思路":
                    st.session_state.genFlag = True
                    st.write("**请分析正确答案并回答以下问题**")

        # Display chat messages from history on app rerun
        if st.session_state.genFlag:
            prompt = st.chat_input("开始输入。。。")

            # Accept user input
            with st.chat_message("assistant"):
                st.title("知识点总结")
                st.write("请详细描述这道题的知识点")

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if prompt:
                if st.session_state.stage == 1:
                    # Display user message in chat message container
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    pickle.dump(st.session_state.messages, open(f"{st.session_state.user}.pkl", "wb"))
                    with st.chat_message("assistant"):
                        response = "bruv"
                        if prompt == "oils":
                            response += "<|done|>"
                        st.write(response.replace("<|done|>", ""))
                        if "<|done|>" in response:
                            st.session_state.stage += 1
                        else:
                            st.session_state.thesis += 1
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.replace("<|done|>", "")})
                    pickle.dump(st.session_state.messages, open(f"{st.session_state.user}.pkl", "wb"))
                if st.session_state.stage == 2:
                    with st.chat_message("assistant"):
                        st.title("考点总结")
                        st.write("请详细描述这道题的考点（知识点的拓展)")
                        # Display user message in chat message container
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    pickle.dump(st.session_state.messages, open(f"{st.session_state.user}.pkl", "wb"))
                    with st.chat_message("assistant"):
                        response = "bruv"
                        if prompt == "oils":
                            response += "<|done|>"
                        st.write(response.replace("<|done|>", ""))
                        if "<|done|>" in response:
                            st.session_state.stage += 1
                        else:
                            st.session_state.test += 1
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.replace("<|done|>", "")})
                    pickle.dump(st.session_state.messages, open(f"{st.session_state.user}.pkl", "wb"))
                if st.session_state.stage == 3:
                    with st.chat_message("assistant"):
                        st.title("下次如何避免出错")
                        st.write("根据以上提供的答案，描述如和下次同样考点不出错")
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    pickle.dump(st.session_state.messages, open(f"{st.session_state.user}.pkl", "wb"))
                    with st.chat_message("assistant"):
                        response = "bruv"
                        if prompt == "oils":
                            response += "<|done|>"
                        st.write(response.replace("<|done|>", ""))
                        if "<|done|>" in response:
                            st.session_state.stage += 1
                        else:
                            st.session_state.prevention += 1

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.replace("<|done|>", "")})
                    pickle.dump(st.session_state.messages, open(f"{st.session_state.user}.pkl", "wb"))
                if st.session_state.stage == 4:
                    with st.chat_message("assistant"):
                        st.title("题型归纳")
                        st.write("总结一下这个错题属于什么样的题型（根据题来尝试总结或取个名字）")
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    pickle.dump(st.session_state.messages, open(f"{st.session_state.user}.pkl", "wb"))
                    with st.chat_message("assistant"):
                        response = "bruv"
                        if prompt == "oils":
                            response += "<|done|>"
                        st.write(response.replace("<|done|>", ""))
                        if "<|done|>" in response:
                            st.session_state.stage += 1
                        else:
                            st.session_state.prevention += 1
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.replace("<|done|>", "")})
                    pickle.dump(st.session_state.messages, open(f"{st.session_state.user}.pkl", "wb"))
                if st.session_state.stage == 5:
                    if not os.path.isdir(f"{st.session_state.user}_wr"):
                            os.mkdir(f"{st.session_state.user}_wr")
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    cv2.imwrite(datetime.now().strftime("%Y/%m/%d")+".png", image)
                    os.system(f"ls")
                    if not os.path.isdir(f"{st.session_state.user}_res"):
                            os.mkdir(f"{st.session_state.user}_res")
                    with st.chat_message("assistant"):
                        st.title("恭喜你完成错题总结！")
                        if os.path.exists(f"{st.session_state.user}_callnum.pkl"):
                            ((thesis, thesisdt), (test, testdt), (prevention, preventiondt),
                             (qtype, qtypedt)) = pickle.load(
                                open(f"{st.session_state.user}_callnum.pkl", "rb"))
                            thesisdt=st.session_state.thesis-thesis
                            testdt=st.session_state.test-test
                            preventiondt=st.session_state.prevention-prevention
                            qtypedt=st.session_state.qtype-qtype
                        else:
                            ((thesis, thesisdt), (test, testdt), (prevention, preventiondt), (qtype, qtypedt)) = (
                                st.session_state.thesis, st.session_state.thesis), (
                                st.session_state.test, st.session_state.test), (
                                st.session_state.prevention, st.session_state.prevention), (
                                st.session_state.qtype, st.session_state.qtype)
                        st.title("对话数据")
                        st.metric(label="知识点总结引导对话数", value=st.session_state.thesis, delta=thesisdt)
                        st.metric(label="考点总结引导对话数", value=st.session_state.test, delta=testdt)
                        st.metric(label="错题避让引导对话数", value=st.session_state.prevention, delta=preventiondt)
                        st.metric(label="题目类型引导对话数", value=st.session_state.qtype, delta=qtypedt)
                        pickle.dump(((thesis, thesisdt), (test, testdt), (prevention, preventiondt), (qtype, qtypedt)), open(f"{st.session_state.user}_callnum.pkl", "wb"))
                        if os.path.exists(f"{st.session_state.user}_cal.pkl"):
                            array = pickle.load(open(f"{st.session_state.user}_cal.pkl", "rb"))
                            now = datetime.now().strftime("%Y/%m/%d")
                            for m,i in enumerate(array["日期"]):
                                if now == i:
                                    array["打卡状态"][m]="✅"
                            pickle.dump(array, open(f"{st.session_state.user}_cal.pkl", "wb"))
                        st.session_state.messages = []
                        pickle.dump([], open(f"{st.session_state.user}.pkl", "wb"))
                        st.session_state.genFlag=False
                        st.session_state.stage=0

                        st.page_link("calender.py",label="打卡✅")
                if st.session_state.stage == 0:
                    st.progress(100)
                else:
                    st.progress(st.session_state.stage * 20)
