# MODEL ZOO


## 1. Pre-trained Models

| Method | Vision Encoder  | #Images | Dataset   | Pretrained Weights |    Training Logs  |
| :---   | :--- | :--- | :---   |    :----:   |          :---: |
| PTP-BLIP| ViT-B(DeiT) | 4M   | CC3M+COCO+VG+SBU | [link](https://huggingface.co/sail/PTP/blob/main/Pretrain_concated_pred_4m.pth)      | [link](https://huggingface.co/sail/PTP/blob/main/4M_pretrain.txt)  |

## 2. Downstream Model


### 2.1 Captioning
| Method | B@4 | CIDEr | | Config    |
| :---   |  :---   | :---  |    ---: |
| PTP-BLIP| 40.1 | 135.0    | configs/caption_coco.yaml |


### 2.2 Zero-shot Retrieval

<!-- #### 2.2.1 COCO
| Method | I2T@1 | T2I@1 | Model Weight  | Training Logs | Config    |
| :---   |  :---   | :---  | :---   | :---  |          :---: |
| PTP-BLIP| 72.3 | 49.5 | [link](https://huggingface.co/sail/PTP/blob/main/zero_shot_coco_checkpoint_4m.pth)      | [link](https://huggingface.co/sail/PTP/blob/main/4M_ptp_coco_zero_shot.txt)      | configs/retrieval_coco.yaml  | -->


#### 2.2.2 Flickr30K

| Method |  I2T@1 | T2I@1 | Model Weight  | Training Logs | Config    |
| :---   |  :---   | :---  |  :---   | :---  |          :---: |
| PTP-BLIP| 86.4 | 67.0 |  [link](https://huggingface.co/sail/PTP/blob/main/zero_shot_coco_checkpoint_4m.pth)   | [link](https://huggingface.co/sail/PTP/blob/main/4M_ptp_flickr30k_zero_shot.txt)      | configs/retrieval_flickr.yaml  |


### 2.3 Retrieval (Fine-tune)

Tip: Please use as large batch size as possible, we experimentally find that the larger batch size leads to better result for this task. Due to memory limiation, we use batch size 24 rather than 28 in original implmentation.


#### 2.3.1 COCO
| Method |I2T@1 | T2I@1 | | Config    |
| :---   |  :---   | :---  |        :---: |
| PTP-BLIP| 77.6 | 59.4 | configs/retrieval_coco.yaml  |


#### 2.3.2 Flickr30K
| Method |I2T@1 | T2I@1 | Model Weight  | Training Logs | Config    |
| :---   | :---   | :---  |  :---   | :---  |          :---: |
| PTP-BLIP|  96.1 | 84.2 | [link](https://huggingface.co/sail/PTP/blob/main/flickr30k_ft_4m.pth)      | [link](https://huggingface.co/sail/PTP/blob/main/4M_ptp_flickr30k_ft.txt)      | configs/retrieval_flickr.yaml  |

### 2.4 VQA V2

| Method | Test-dev|Test-std |Model Weight  | Training Logs | Config    |
| :---   |  :---   | :---  | :---   | :---  |  :---: |
| PTP-BLIP| 76.02 | 76.18 | [link](https://huggingface.co/sail/PTP/blob/main/vqa_ft_4m.pth)      | [link](https://huggingface.co/sail/PTP/blob/main/4M_ptp_vqa_v2.txt)      | configs/vqa.yaml  |

### 2.5 NLVR

| Method | Dev| Test-P | Model Weight  | Training Logs | Config    |
| :---   |  :---   | :---  | :---   | :---  |          :---: |
| PTP-BLIP| 80.45 | 80.70 | [link](https://huggingface.co/sail/PTP/blob/main/nlvr_ft_4m.pth)      | [link](https://huggingface.co/sail/PTP/blob/main/4M_ptp_nlvr.txt)      | configs/nlvr.yaml  |