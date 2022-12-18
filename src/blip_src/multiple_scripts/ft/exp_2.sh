# image captioning

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}


echo '(coco captioning:) load pretrained model from: '$3;
sed -i "/^\(pretrained: \).*/s//\1'$3'/" ./configs/caption_coco.yaml;

# step1: zero-shot coco captioning
echo 'step1: zero-shot coco captioning'
python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$(rand 2000 4000)  train_caption.py \
--config ./configs/caption_coco.yaml \
--output_dir output/captioning_coco_$2 --evaluate

# step2: fine-tune

echo 'step2: fine-tune coco captioning'
python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$(rand 2000 4000)  train_caption.py \
--config ./configs/caption_coco.yaml \
--output_dir output/captioning_coco_$2 # --evaluate