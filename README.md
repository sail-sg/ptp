# PTP

## Code for the paper: Position-guided Text Prompt for Vision Language Pre-training [Arxiv]()

## Introduction
Position-guided Text Prompt (PTP) aims to bring position information into conventional Vision-Language Pre-training (VLP) models, as current mainstream e2e VLP models ignore this important cues.

**Position information is missed in a well-trained ViLT models (B).**
![motivation](imgs/motivation.png)


![main_ppl](imgs/main_ppl-2.png)

Our method provide a good altentive for existing object feature based method.

## Updates

- We have put the pretrained weight on hugging face for fast download.
- The first version of downstream evaluation code is released! The pre-training code is in cleaning up now.



##  Installation

Please find installation instructions for PyTorch in [INSTALL.md](INSTALL.md).


## Dataset Preparation

You may follow the instructions in [DATASET.md](DATASET.md) to prepare the datasets.
Considering the dataset prepartion is very time consuming, we provide detail guidence and provided our trained corpus.


## Model Zoo
We provide pre-trained models and fine-tune models in [MODEL_ZOO.md](MODEL_ZOO.md).


## Quick Start

Follow the example in [GETTING_STARTED.md](GETTING_STARTED.md) to start playing vlp models with PTP.

## Transfer To Other Architectures

The PTP can easily transfer to other architectures without much effort. 
Specifically, PTP include two steps:

- Download or generate corpus as ours.
- Modify the dataset.py

Then train the model with original objectives.

## Ackowledgement
This work is mainly based on [BLIP](https://github.com/salesforce/BLIP), thanks for this good baseline. 
We also refer [ViLT](https://github.com/dandelin/ViLT) and [OSCAR](https://github.com/microsoft/Oscar) for ablation study and dataset preparation.

## License
PTP is released under the Apache 2.0 license.

## Contact

Email: awinyimgprocess at gmail dot com

If you have any questions, please email me or open an new issue.

## Citation
If you find our work helps, please use the following BibTeX entry for citation.

```
@article{wang2022ptp,
  title={Position-guided Text Prompt for Vision Language Pre-training},
  author={Wang, Alex Jinpeng and Zhou, Pan and Shou, Mike Zheng and Yan, Shui Cheng},
  journal={arXiv preprint arXiv:},
  year={2022}
}
```