import os
import streamlit as st
import time
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import time
import cv2
import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from models import *
base = ""
def stackPDF(image_paths,output):
    c = canvas.Canvas(output, pagesize=A4)
    # Load images and get their dimensions
    images = [Image.open(img_path) for img_path in image_paths]
    page_width, page_height = A4
    current_height = page_height

    # Draw images onto canvas
    for img in images:
        img_width, img_height = img.size
        new_width = page_width
        new_height = (page_width / img_width) * img_height

        # Check if there is enough space left on the current page
        if current_height - new_height < 0:
            c.showPage()
            current_height = page_height

        c.drawImage(img.filename, 0, current_height - new_height, width=new_width, height=new_height)
        current_height -= new_height
    c.save()
result_path=f"{st.session_state.user}_res/"  #The folder path that you want to save the results
stackPDF(st.session_state.review_list, f"{st.session_state.user}.pdf")
with open(f"{st.session_state.user}.pdf", 'rb') as file:
    file_contents = file.read()

# Create a downloadable link for the PDF
st.download_button(
    label="下载复习试卷",
    data=file_contents,
    file_name="复习试卷.pdf",
    mime="application/pdf"
)
