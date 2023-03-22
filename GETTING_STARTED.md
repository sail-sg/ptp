# Getting Started with PTP Model

## PTP-BLIP

### 1. Download VIT-Base Models.

```bash
cd ptp/src/blip_src
mkdir pretrained_models && cd pretrained_models;
wget -c https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth;
```

### 2. Pre-train

```bash
python move_pretrained_weights.py;

python -m torch.distributed.launch --nproc_per_node=8 pretrain_concated_pred_tsv.py \
--config ./configs/mt_pt/tsv/pretrain_concated_pred_4M.yaml --output_dir output/Pretrain_concated_pred_4M

echo "output dir is: output/Pretrain_concated_pred_4M"

```

Alternatively, download our pretrained model from [MODEL_ZOO.md](MODEL_ZOO.md).

### 3. Downstream Task Evaluation
After pre-trained, replace the **pretrained:** in yaml of each task with pre-trained model or downloaded model.
Then we provide run scripts for these tasks:

#### Captioning

```bash
# image captioning

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}


echo '(coco captioning:) load pretrained model from: ';

python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$(rand 2000 4000)  train_caption.py \
--config ./configs/caption_coco.yaml \
--output_dir output/captioning_coco
```

#### Retrieval

```bash
function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}


# ============================== step1: zero-shot retrieval evaluation ========= 
# evaluate on test 
echo '(coco step1: zero-shot evaluation) load pretrained model from: ';

python -m torch.distributed.launch --nproc_per_node=$1  --master_port=$(rand 2000 4000) train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco_zs \
--evaluate

# ================= step2: train and evaluate val & test set =================
echo '(coco step2: train on retrieval) load pretrained model from: ';

python -m torch.distributed.launch --nproc_per_node=$1  --master_port=$(rand 2000 4000) train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco_ft

```

#### VQA


```bash
# vqa

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}


echo '(vqa:) load pretrained model from: ';

python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$(rand 2000 4000) \
train_vqa.py --config ./configs/vqa.yaml \
--output_dir output/vqa_v2_vqa 
```

After generate result files, submitted in [eval_ai](https://eval.ai/web/challenges/challenge-page/830) for final results.


#### NLVR

```bash
# NLVR2

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}

echo '(NLVR:) load pretrained model from: ';

python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$(rand 2000 4000) train_nlvr.py \
--config ./configs/nlvr.yaml \
--output_dir output/NLVR_NLVR
```


#### Run All Downstream Task At Once
We also provide a shell script for all downstream task as below:

```bash
bash multiple_scripts/multiple_exp_all_single_8u_ft.sh Pretrain_concated_pred_4M
```

where _Pretrain_concated_pred_4M_ is the pretrained output directory.

The
```bash
python move_pretrained_weights.py;

gpu_num=8;
time=$(date "+%Y-%m-%d-%H:%M:%S");
suffix=$1${time}; # the suffix to distingush different experiment, e.g. $1='generation_mix'

PRETRAINED_MODEL="output\/$1\/checkpoint_19.pth"

echo "${suffix}";

bash multiple_scripts/ft/exp_2.sh $gpu_num $suffix $PRETRAINED_MODEL; # captioning, ~1h

bash multiple_scripts/ft/exp_5.sh $gpu_num $suffix $PRETRAINED_MODEL; # flickr30 retrieval, ~1h

bash multiple_scripts/ft/exp_4.sh $gpu_num $suffix $PRETRAINED_MODEL; # NLVR2, ~2h

bash multiple_scripts/ft/exp_1.sh $gpu_num $suffix $PRETRAINED_MODEL; # coco retrieval, ~12h

bash multiple_scripts/ft/exp_3.sh $gpu_num $suffix $PRETRAINED_MODEL; # vqa, very slow, ~35 h


```

**Tip**
The simplest way to evaluate model on all tasks is:
```bash
bash multiple_scripts/multiple_exp_all_single_8u_ft.sh
```


## PTP-ViLT

### 1. Pre-train
```bash
python run.py with data_root=/dataset num_gpus=8 num_nodes=1 task_mlm_itm whole_word_masking=True step200k per_gpu_batchsize=64
```

### 2. Downstream Tasks Evaluation

#### vqa
```bash
python run.py with data_root=/dataset num_gpus=8 num_nodes=1 per_gpu_batchsize=64 task_finetune_vqa_randaug test_only=True precision=32 load_path="weights/vilt_vqa.ckpt"

```