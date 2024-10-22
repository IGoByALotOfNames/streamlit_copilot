import pandas as pd
from datetime import datetime,timedelta
import streamlit as st
import pickle
import numpy as np
import os
#inits = {"日期": [(datetime.now()+timedelta(days=i)).strftime("%Y/%m/%d") for i in range(100)], "打卡状态":["⬜" for _ in range(100)]}
#pickle.dump(inits,open("{st.session_state.user}_cal.pkl","wb"))
col = st.columns((3, 6,4), gap='medium')
if os.path.exists(f"{st.session_state.user}_callnum.pkl"):
  ((thesis, thesisdt), (test, testdt), (prevention, preventiondt), (qtype, qtypedt)) = pickle.load(
    open(f"{st.session_state.user}_callnum.pkl", "rb"))
  thesis += thesisdt
  test += testdt
  prevention += preventiondt
  qtype += qtypedt

else:
  ((thesis, thesisdt), (test, testdt), (prevention, preventiondt), (qtype, qtypedt)) = (0, 0), (0, 0), (0, 0), (0, 0)
#os.system("ls")
st.session_state.progress=0
with col[0]:
  st.title("对话数据")
  st.metric(label="知识点总结引导对话数", value=thesis, delta=thesisdt)
  st.metric(label="考点总结引导对话数", value=test, delta=testdt)
  st.metric(label="错题避让引导对话数", value=prevention, delta=preventiondt)
  st.metric(label="题目类型引导对话数", value=qtype, delta=qtypedt)
with col[2]:
  if os.path.exists(f"{st.session_state.user}_cal.pkl"):
    array = pickle.load(open(f"{st.session_state.user}_cal.pkl", "rb"))
    for x in array['打卡状态']:
      if x == "✅":
        st.session_state.progress+=1
    st.title("打卡日历")
    df = pd.DataFrame(array)
    st.dataframe(df) 
  else:
    st.write("还没有布置打卡呢")
with col[1]:
  # For example, 70%
  # Define colors for each 10% increment
  colors = [
    "#7DF9FF",  "#DC143C","#FFF700",  "#8A2BE2",
    "#FFA500", "#228B22","#FF7F50", "#FF69B4", " #40E0D0", "#4B0082",
  ]


  # Function to determine the color based on progress
  def get_color(percentage):
    return ",".join(colors[0:int(percentage // 10)])
  # HTML and CSS for the jar progress bar
  st.title("知识罐")
  jar_html = f"""
  <div style="width: 205px; height: 30px; border: 2px solid #000; position: relative; margin: auto; border-radius: 0px;background-color:black;border-radius: 5px;"> 
  <div style="width: 200px; height: 350px; border: 2px solid #000; position: relative; margin: auto; border-radius: 20px; top:20px;">
      <div class = "bg" style="width: 100%; height: {st.session_state.progress}%;
background-image: linear-gradient(to top, {get_color(st.session_state.progress)}); position: absolute; bottom: 0; border-radius: 15px;"></div>
      <div style="position: absolute; bottom: 30px; width: 100%; text-align: center;z-index:999;">第10天</div>
      <div style="position: absolute; bottom: 60px; width: 100%; text-align: center;z-index:999;">第20天</div>
      <div style="position: absolute; bottom: 90px; width: 100%; text-align: center;z-index:999;">第30天</div>
      <div style="position: absolute; bottom: 120px; width: 100%; text-align: center;z-index:999;">第40天</div>
      <div style="position: absolute; bottom: 150px; width: 100%; text-align: center;z-index:999;">第50天</div>
      <div style="position: absolute; bottom: 180px; width: 100%; text-align: center;z-index:999;">第60天</div>
      <div style="position: absolute; bottom: 210px; width: 100%; text-align: center;z-index:999;">第70天</div>
      <div style="position: absolute; bottom: 240px; width: 100%; text-align: center;z-index:999;">第80天</div>
      <div style="position: absolute; bottom: 270px; width: 100%; text-align: center;z-index:999;">第90天</div>
      <div style="position: absolute; bottom: 300px; width: 100%; text-align: center;z-index:999;">第100天</div>
  </div>
  """

  # Display the progress bar in Streamlit
  st.write(jar_html, unsafe_allow_html=True)
  st.write(f"<p style='top:378px;text-align:center;position: relative;'>已完成{st.session_state.progress/100}%</p>",unsafe_allow_html=True)

