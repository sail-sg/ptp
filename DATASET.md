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

## 2. Download Corpus (image-text pair)
We provide three kinds of shuffled image-text pair. We use object information from [OSCAR](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md) and follow [BLIP](https://github.com/salesforce/BLIP) for caption clean.

- 3M Image: [CC3M]()
- 4M Image: [CC3M+COCO+VG+SBU]()
- 14M Image: [CC12M+CC3M+COCO+VG+SBU]()
