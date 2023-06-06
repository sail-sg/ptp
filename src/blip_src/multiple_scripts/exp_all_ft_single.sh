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
