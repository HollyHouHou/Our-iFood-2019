#!/usr/bin/env python
# coding: utf-8

# In[95]:


from classification_models.resnet import ResNet18, preprocess_input
import numpy as np
import pandas as pd
import string as str
from skimage.transform import resize
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread,imshow
import os
model = ResNet18((224, 224, 3), weights='imagenet')


# In[96]:


def pre_process_pic(x):
    x=resize(x, (224, 224))*255
    x = preprocess_input(x)
    x = np.expand_dims(x, 0)
    return x

def judge_label(predicts,target_list):
    predicts=predicts[0]
    for predict in predicts:
        if(predict[0] in target_list):
            return True
    return False


# In[98]:


col = ["label"]
data = pd.read_csv("ImageNet_Select_Food.txt", header = None, names = col )
food_list = []
for i in range(156):
    food_list.append(data["label"][i])


rootdir='./samples/'#图片文件夹路径
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])#path即为图片路径
    if os.path.isfile(path):
        x = imread(path)
        imshow(x)
        x = pre_process_pic(x)
        y = model.predict(x)
        print(path)
        print(decode_predictions(y))
        tmp = judge_label(decode_predictions(y),food_list)
        print(tmp)
        print("\n")


# In[ ]:




