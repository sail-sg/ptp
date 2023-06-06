import json
import os
import random
import pandas as pd
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
from PIL import ImageFile
from PIL.Image import blend as blend
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import ast
from data.utils import pre_caption
import os,glob

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

        
    def generate_bbox_annotation(self, img, ann):
        prompt_text = '.'
        if len(ann) > 2:
            ann[2] = json.loads(ann[2])
            ann[3] = ast.literal_eval(ann[3])
            object_num = len(ann[3])
            if object_num > 0:
                sample_index = random.randint(0, object_num-1)
                w, h = img.size
                bbox_loc = ann[2][sample_index]
                # print(bbox_loc)
                w_1 = min(int((bbox_loc[0]/2 + bbox_loc[2]/2)/w * 3), 2)
                h_1 = min(int((bbox_loc[1]/2 + bbox_loc[3]/2)/h * 3), 2)
                # print(w_1, h_1, w, h)
                block = 3*h_1 + w_1
                prompt_text = '. The block ' + str(block) + ' has a ' + ann[3][sample_index] + ' .';
            else:
                prompt_text = "."
        return prompt_text

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):    
        ann = self.annotation.iloc[index]
        try:
            image = Image.open(os.path.join(self.img_root, ann[0])).convert('RGB')  
            caption_str = ann[1]
            if caption_str[0] == '[':
                captions = ast.literal_eval(caption_str)
                temp_caption = captions[-1] + captions[random.randint(0, len(captions) - 2)]
            else:
                temp_caption = caption_str 
            temp_caption = temp_caption + self.generate_bbox_annotation(image, ann)
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, self.__len__()-1))
        image = self.transform(image)
        caption = pre_caption(temp_caption, 30)        
        # caption = ann['caption']
        return image, caption

    # def __getitem__(self, index):    
    #     ann = self.annotation.iloc[index]
    #     try:
    #         image = Image.open(os.path.join(self.img_root, ann[0])).convert('RGB')  
    #         caption_str = ann[1]
    #         if caption_str[0] == '[':
    #             captions = ast.literal_eval(caption_str)
    #             if len(captions) < 6:
    #                 temp_caption = captions[random.randint(0, len(captions) - 1)]
    #             else:
    #                 temp_caption = captions[random.randint(0, len(captions)//3 - 1)] + ', ' + \
    #                 captions[random.randint(len(captions)//3, len(captions)//3 * 2)] + ', ' + \
    #                 captions[random.randint(len(captions)//3*2, len(captions)-1)]
    #         else:
    #             temp_caption = caption_str 
    #         temp_caption = temp_caption + self.generate_bbox_annotation(image, ann)
    #     except Exception as e:
    #         print(e)
    #         return self.__getitem__(random.randint(0, self.__len__()-1))
    #     image = self.transform(image)
    #     caption = pre_caption(temp_caption, 30)        
    #     # caption = ann['caption']
    #     return image, caption