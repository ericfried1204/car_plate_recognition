import warnings
warnings.filterwarnings("ignore")
import pdb
import tensorflow as tf
import pandas as pd
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import os
import shutil
import re
import glob
import subprocess
import random
import yaml
import tqdm
import gc



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

import IPython.display as display
from IPython.display import Video
from PIL import Image
import cv2

import ultralytics
from ultralytics import YOLO

import xml.etree.ElementTree as xet
from bs4 import BeautifulSoup

import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import pytesseract
import easyocr
# print('ultralytics version: ',ultralytics.__version__)

char_list = 'O045LC2T71Q83RG6BA9DWEYFNMSHX'

class CFG:
    
    out_folder = f'./kaggle/working'
    class_name = ['car_plate']
    video_test_path = 'https://docs.google.com/uc?export=download&confirm=&id=1pz68D1Gsx80MoPg-_q-IbEdESEmyVLm-'

    weights = 'best.pt'
    exp_name = 'car_plate_detection'
    img_size = (240,400)
    
    epochs = 50
    batch_size = 16

    optimizer = 'auto' # SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto
    lr = 1e-3
    lr_factor = 0.01 #lo*lr_f
    weight_decay = 5e-4
    dropout = 0.0
    patience = int(0.7*epochs)
    profile = False
    label_smoothing = 0.0 
    seed=42
    
    confidance = 0.5

 

def display_image(image, print_info = True, hide_axis = False):
    if image is None:
        print("No image to display.")
        return
    if isinstance(image, str):  # Check if it's a file path
        img = Image.open(image)
        fig = plt.figure(figsize = (15,15))
        plt.imshow(img)
    elif isinstance(image, np.ndarray):  # Check if it's a NumPy array
        image = image[..., ::-1]  # BGR to RGB
        img = Image.fromarray(image)
        plt.imshow(img)
    else:
        raise ValueError("Unsupported image format")

    if print_info:
        print('Type: ', type(img), '\n')
        print('Shape: ', np.array(img).shape, '\n')

    if hide_axis:
        plt.axis('off')

    plt.show()


        
def free_gpu_cache() -> None:
    print("Initial GPU Usage")
    gpu_usage()


    torch.cuda.empty_cache()

#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


model = YOLO(CFG.weights)

def extract_roi(image, bounding_box):

    x_min, x_max, y_min, y_max = bounding_box
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image
        
# os.path.join(CFG.out_folder, 'valid')+'/*.jpg'

valid_images = glob.glob(f'./test_images/*.jpeg')
valid_images+= glob.glob(f'./test_images/*.jpg')
valid_images+= glob.glob(f'./test_images/*.png')




if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    if num_gpus > 1:
        train_device, test_device = 0,1
      
    else:
        train_device, test_device = 0,0        

    
    model.to(f'cuda:{train_device}')

    # Get information about each GPU
    for i in range(num_gpus):
        gpu_properties = torch.cuda.get_device_properties(i)
        # print(f"\nGPU {i}: {gpu_properties.name}")
        # print(f"  Total Memory: {gpu_properties.total_memory / (1024**3):.2f} GB")
        # print(f"  CUDA Version: {gpu_properties.major}.{gpu_properties.minor}")
        

else:
    # print("CUDA is not available. You can only use CPU.")
    train_device, test_device = 'cpu', 'cpu'
    model.to(train_device)
    

# gpu_usage()

# print('Model: ', CFG.weights)
# print('Device: ' ,model.device)
def extract_ocr(img_path, model):
    img = cv2.imread(img_path)
    
    # img = Image.open(img_path)
    if img is None:
        # print(f"Failed to load image: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img)

    results = model.predict(img, 
                        conf = 0.3, 
                        classes =[0], 
                        device = test_device, 
                        verbose = False)

    df_coords = pd.DataFrame(results[0].cpu().numpy().boxes.data, 
                                   columns = ['xmin', 'ymin', 'xmax', 
                                              'ymax', 'conf', 'class'])
    df_coords['class'] = df_coords['class'].replace({0:'car_plate'})

    bboxs = df_coords[['xmin','xmax','ymin','ymax']].values.astype(int)
    # print(bboxs)
    classes = df_coords['class'].values

    df_plate = pd.DataFrame()
    roi_img = None
    for i,bbox in enumerate(bboxs):        
        roi_img = extract_roi(img, bbox)

    return roi_img

def decode_char(txt):
    charlist = ""
    for index, char in enumerate(txt):
        try:
            if(char=='O'):
                charlist+="ه"
            elif(char=='L'):
                charlist+="ل"
            elif(char=='C'):
                charlist+="س"
            elif(char=='T'):
                charlist+="ط"
            elif(char=='Q'):
                charlist+="ق"
            elif(char=='R'):
                charlist+="ر"
            elif(char=='G'):
                charlist+="ج"
            elif(char=='B'):
                charlist+="ب"
            elif(char=='A'):
                charlist+="ا"
            elif(char=='D'):
                charlist+=" "  
            elif(char=='د'):
                charlist+=" " 
            elif(char=='W'):
                charlist+=" " 
            elif(char=='و'):
                charlist+=" " 
            elif(char=='E'):
                charlist+="ع"  
            elif(char=='Y'):
                charlist+="ى " 
            elif(char=='F'):
                charlist+="ف" 
            elif(char=='N'):
                charlist+="ن" 
            elif(char=='M'):
                charlist+="م" 
            elif(char=='S'):
                charlist+="ص" 
            elif(char=='H'):
                charlist+="ح" 
            elif(char=='X'):
                charlist+="" 
            else:
                charlist+=char
        except:
            print(char)
    return charlist



inputs = Input(shape=(32,128,1))
 
# VGG
conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
# Batch normalization layer
batch_norm_1 = BatchNormalization()(conv_1)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(batch_norm_1)
 
conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
# Batch normalization layer
batch_norm_2 = BatchNormalization()(conv_2)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(batch_norm_2)
 
conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
 
conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
# poolig layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)
 
conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
 
conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
 
# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)

act_model.load_weights('final_model_last.hdf5')

test_img=[]
names = []

for path in valid_images:
    # new_image = extract_ocr(path, model)
    # display_image(new_image)

    img = extract_ocr(path, model)
    if img is None:
        continue
   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 32))
    
    img = np.expand_dims(img , axis = 2)
    test_img.append(img)
    names.append(path)
test_img = np.array(test_img)
prediction = act_model.predict(test_img[:])
 
# use CTC decoder
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])
 
# see the results
i = 0
for x in out:
    print("original_text =  ", names[i])
    print("predicted text = ", end = '')    
    s=""
    for p in x:  
        if int(p) != -1:
            s+=char_list[int(p)]                   
    i+=1
    print(decode_char(s))
    print("\n")

input("Press Enter to exit")