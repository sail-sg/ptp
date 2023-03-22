# augment the boundng box with the original images
import json
import os
import random
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
from PIL import ImageFile
from PIL.Image import blend as blend
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import PIL
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from data.utils import pre_caption
import os,glob
import math
import cv2
import ast
import gc

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, laion_path, transform): 
        self.img_root = "/dataset" # server
        self.ann_pretrain = None
        for f in ann_file:
            ann_temp = pd.read_csv(f, sep='\t', header=None)
            if self.ann_pretrain is None:
                self.ann_pretrain = ann_temp
            else:
                self.ann_pretrain = pd.concat([self.ann_pretrain, ann_temp], ignore_index=True, sort=False)
        
        self.annotation = self.ann_pretrain
            
        self.transform = transform
        self.normalize = torchvision.transforms.Compose([torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


    def reload_laion(self, epoch):
        n = epoch%len(self.laion_files)
        print('loading '+self.laion_files[n])
        with open(self.laion_files[n],'r') as f:
            self.ann_laion = json.load(f)      
        
        self.annotation = self.ann_pretrain + self.ann_laion    
        
    def __len__(self):
        return len(self.annotation)
    
    def generate_bbox_img(self, img, ann):
        object_num = len(ann[3])
        sample_index = random.randint(0, object_num-1)
        object_tag = ann[3][sample_index]
        w, h = img.size
        bbox_loc = ann[2][sample_index]
        im = PIL.Image.new(mode="RGB", size=(w, h))
        # im = np.asarray(im)
        im = np.array(im)
        im[bbox_loc[1]:bbox_loc[3], bbox_loc[0]:bbox_loc[2]] = 255
        im = Image.fromarray(im)
        return im, object_tag

    def find_squares(self, image):
        square_exist = False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)  
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
        # print(len(contours))
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0]) 
            square_exist = True
        else:
            x, y, w, h = 0, 0, image.shape[0], image.shape[1]
        return [x, y, w, h], square_exist

    def generate_coord_info(self, bbox_img):
        bbox_img = torchvision.transforms.ToPILImage()(bbox_img)
        bbox_img = np.asarray(bbox_img)
        h, w = bbox_img.shape[:2]
        [x_b, y_b, w_b, h_b], square_exist = self.find_squares(bbox_img)
        cv2.rectangle(bbox_img, (x_b, y_b), (x_b+w_b, y_b+h_b), (0, 255, 0), 2)
        # x_b = int(x_b/h*100)
        # y_b = int(y_b/w*100)
        # w_b = int(w_b/h*100)
        # h_b = int(h_b/w*100)
        return [x_b, y_b, w_b, h_b], [h, w], square_exist

    def __getitem__(self, index):           
        ann = self.annotation.iloc[index]
        try:   
            image = Image.open(os.path.join(self.img_root, ann[0])).convert('RGB')   
            if len(ann.keys()) > 2:
                ann[2] = json.loads(ann[2])
                ann[3] = ast.literal_eval(ann[3])
                if len(ann[2]) > 0:
                    # w, h = img.size # original img size
                    # original_image = np.asarray(image)
                    bbox_img, object_tag = self.generate_bbox_img(image, ann)
                    seed = np.random.randint(2147483647) # make a seed with numpy generator 
                    random.seed(seed) # apply this seed to img tranfsorms
                    torch.manual_seed(seed) # needed for torchvision 0.7
                    np.random.seed(seed)
                    image = self.transform(image)
                    # original_bbox_image = np.asarray(bbox_img)
                    # step3: transform bbox image and getting block information
                    random.seed(seed) # apply this seed to target tranfsorms
                    torch.manual_seed(seed) # needed for torchvision 0.7
                    np.random.seed(seed)
                    bbox_img = self.transform(bbox_img)
                    [x_b, y_b, w_b, h_b], [h, w], square_exist = self.generate_coord_info(bbox_img)

                    # prevent memory leverage
                    del bbox_img
                    gc.collect()

                    # step4: get the block_index, use x to mean there is no box or all the figure
                    if square_exist == False:
                        block = 'x'
                    else:
                        w_1 = min(int((x_b + w_b/2)/w * 3), 2)
                        h_1 = min(int((y_b + h_b/2)/h * 3), 2)
                        # print(w_1, h_1, w, h)
                        block = 3*h_1 + w_1
                    prompt_text = '. The block ' + str(block) + ' has a ' + object_tag + '.'
                    ann[1] = pre_caption(ann[1], 22) + '. ' + pre_caption(prompt_text, 8)
                else:
                    image = self.transform(image)
                    ann[1] = pre_caption(ann[1], 30)
            else:
                image = self.transform(image)
                ann[1] = pre_caption(ann[1], 30)
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, self.__len__()-1))
        caption = ann[1]
        # print(caption)
        image = self.normalize(image)
        return image, caption