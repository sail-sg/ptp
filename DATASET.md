# Datasets
We prepare the pre-training corpus following OSCAR and BLIP.
__As the data prepartion is very time consuming, we provide our experience for reference.__

## 1. Download Datasets (images)
### Pre-train Datasets:

### CC3M
Step1: First download train/val/test annotation files include URL from [google-research-datasets](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md).

Step2: We provided our script for downloading and split CC3M into subsplit in [cc3m_download.py](https://huggingface.co/sail/PTP/blob/main/download_cc3m.py).
**It's better to use our cript for downloading as the filename maybe different with different preprocess.**

Notice we only download 2.8M data as some URLs has invalid.

### SBU
First from annotation files include URL from [huggingface](https://huggingface.co/datasets/sbu_captions).

Tip: We provided our script for downloading sbu:
[download_sbu.py](https://huggingface.co/sail/PTP/blob/main/download_sbu.py)

### Visual Genome

Download image (version1.2) from [visualgenome](https://visualgenome.org/api/v0/api_home.html).

The download dirs will be VG_100K and VG_100K_2.
```bash
mkdir image
mv VG_100K/* image/
mv VG_100K_2/* image/
```

### COCO

Down image (coco2014) from [coco](https://cocodataset.org/#download).
Download 2014 Train, 2014 val and 2015 Test images.

### CC12M
Step1: Download annotation files include URLs from [google-research-datasets](https://github.com/google-research-datasets/conceptual-12m).

Step2: Just modify the source tsv file and image path in cc3m_download.py. Then download data the same as cc3m.

Notice we only download 10M data as some URLs has invalid.

### Fine-tune Datasets:

### COCO
Down image (coco2014) from [coco](https://cocodataset.org/#download).
Download 2014 Train, 2014 val, 2014 test and 2015 Test images.


### Flickr30K
Download image from [kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).

### VQA V2

Download images from [VQA](https://visualqa.org/download.html).

### NLVR
Download images from [NLVR](https://lil.nlp.cornell.edu/nlvr/).

## Originze Datasets

Prepare the datasets as follow:
```
Dataset/
    CC3M/
        images/
            train/x/*.jpg
            val/x/*.jpg
    SBU/
        dataset/
            train/x/*.png
    coco2014/
        COCO2014/
            train2014/*.jpg
            val2014/*.jpg
            test2015/*.jpg
    
    VisualGenome/
        image/*.jpg
```

Use soft link to map directory, for example
```bash
ln -s [PATH_TO_COCO2014] Dataset/coco2014/COCO2014
```

## 2. Download/Prepare Corpus (image-text pair)
We provide two kinds of shuffled image-text pair. We use object information from [OSCAR](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md) and follow [BLIP](https://github.com/salesforce/BLIP) for caption refine.
1. Specifically, we download corups and object features from OSCAR codebase first. Follow [download_cc3m_predictions.sh](src/data_preprocess/download_cc3m_predictions.sh) for details. Download COCOTrain, CC Train, SBU (all) and VG.
2. Then Generate object_bbox and object_classes from object feature. Follow [generate_sample_with_bbox_and_classes.py](src/data_preprocess/generate_sample_with_bbox_and_classes.py) for details.
3. At last, use generated caption to padding with origing caption, follow BLIP.

**Notice each COCO image include 5 text in [oscar corpus](https://biglmdiag.blob.core.windows.net/vinvl/pretrain_corpus/coco_flickr30k_gqa.tsv). As COCO is high-quality caption, it will affect the final downstream result much.**

Make sure each line in corpus is
```
[image, refined_caption, object_bbox, object_classes]
```
A example is given below:

```bash
CC3M/images/train/1597/3250687125.jpg   i shall be bringing this white chair and table to the shoot; a white table with two white chairs and a couch    [[340, 226, 417, 323], [16, 364, 348, 810], [256, 206, 380, 325], [195, 322, 627, 899], [0, 0, 192, 288], [568, 198, 730, 335], [95, 107, 202, 141], [531, 0, 732, 191], [666, 244, 734, 369], [378, 208, 677, 341]] ['pillow', 'chair', 'pillow', 'table', 'window', 'pillow', 'box', 'window', 'pillow', 'pillow']
```

- 2.8M Image (2G): [CC3M](https://drive.google.com/file/d/1iO-d5e7mOvWEreDrlNyEc_RU_gP7FNBk/view?usp=sharing)

The filtered file path is:

- 4M Image (2.38G): [CC3M+COCO+VG+SBU](https://drive.google.com/file/d/1NnI-_ha4oqeZeHVOv1GBcvV1txgO9R68/view?usp=sharing)

Thanks Jaeseok Byun for helping correct this corpus.

As we used all spaces for huggingface and google driver now, follow mentonied way to prepare more large corpus.