import os
import streamlit as st
import time
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from skimage import io
import time
from glob import glob
from tqdm import tqdm
import cv2
import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from models import *
base = "D:/projmom/"
def stackPDF(image_paths,output):
    c = canvas.Canvas(output, pagesize=A4)
    # Load images and get their dimensions
    images = [Image.open(image_paths+img_path) for img_path in os.listdir(image_paths)]
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
dataset_path=base+f"{st.session_state.user}_wr"  #Your dataset path
model_path=base+"IS-Net/isnet.pth"  # the model path
result_path=base+f"{st.session_state.user}_res/"  #The folder path that you want to save the results
input_size = [1024, 1024]
net = ISNetDIS()
net.load_state_dict(torch.load(model_path, map_location="cpu"))
net.eval()
im_list = glob(dataset_path + "/*.jpg") + glob(dataset_path + "/*.JPG") + glob(dataset_path + "/*.jpeg") + glob(
    dataset_path + "/*.JPEG") + glob(dataset_path + "/*.png") + glob(dataset_path + "/*.PNG") + glob(
    dataset_path + "/*.bmp") + glob(dataset_path + "/*.BMP") + glob(dataset_path + "/*.tiff") + glob(
    dataset_path + "/*.TIFF")
with torch.no_grad():
    for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
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
        cv2.imwrite(base+result_path + im_name + "_background.png", cropped)
stackPDF(base+result_path, base+f"{st.session_state.user}.pdf")
with open(base+f"{st.session_state.user}.pdf", 'rb') as file:
    file_contents = file.read()

# Create a downloadable link for the PDF
st.download_button(
    label="下载复习试卷",
    data=file_contents,
    file_name="复习试卷.pdf",
    mime="application/pdf"
)