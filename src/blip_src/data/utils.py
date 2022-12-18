import re
import json
import os

import torch
import torch.distributed as dist
import random
import utils
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import cv2

colormaps = [(255, 0, 0), (0, 255, 0), (0, 0, 255),  (61, 145, 64),  (127, 255, 212), (0, 201, 87),
(218, 112, 214), (255, 0, 255), (112, 128, 105), (250, 235, 215),
(240, 255, 255), (252, 230, 201), (255, 255, 0), (235, 142, 85),
(255, 97, 0), (176, 224, 230), (65, 106, 225,), (0, 255, 255),
(56, 94, 15), (8, 46, 84), (255, 192, 203)]

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question

def draw_bboxs(image, bboxs):
    image_w_box = ImageDraw.Draw(image) 
    for bbox in bboxs:
        image_w_box.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], fill=False, outline="red", width=4)
    return image

def draw_bboxs_color_prompt(image, bboxs):
    img = np.asarray(image)
    for index, bbox in enumerate(bboxs):
        sub_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        white_rect[:, :, 0] = colormaps[index][0]
        white_rect[:, :, 1] = colormaps[index][1]
        white_rect[:, :, 2] = colormaps[index][2]
        res = cv2.addWeighted(sub_img, 0.7, white_rect, 0.3, 1.0)
        img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = res
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colormaps[index], 3)
    img = Image.fromarray(img.astype('uint8'))
    return image

def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file



from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval