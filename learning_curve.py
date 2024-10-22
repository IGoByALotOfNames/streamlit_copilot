# Imports for all of the code
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time

def f(t):
    return 184/(np.log(t+1)**1.25 + 1.84)
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

# How to set the graph size 
two_subplot_fig = plt.figure(figsize=(6,6))
plt.subplot(211)
plt.plot(t1, f(t1), color='tab:blue', marker=',')
#point = plt.plot(1,f(1,1,1), 'o',color="red")
st.write("记忆(%)")
st.pyplot(two_subplot_fig)
st.markdown("<p style='text-align: center;'>天数</p>", unsafe_allow_html=True)
