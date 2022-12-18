# image to text retrieval

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}

# ===================== step1: zero-shot evaluation================
echo '(step1: zero-shot f30k retrieval:) load pretrained model from: '$3;
sed -i "/^\(pretrained: \).*/s//\1'$3'/" ./configs/retrieval_flickr.yaml;

python -m torch.distributed.launch --nproc_per_node=$1  --master_port=$(rand 2000 4000) train_retrieval.py \
--config ./configs/retrieval_flickr.yaml \
--output_dir output/retrieval_flickr_$2 \
--evaluate


# ===================== step2: ft and evaluate ================
echo '(step2: fine-tune f30k retrieval:) load pretrained model from: '$3;
sed -i "/^\(pretrained: \).*/s//\1'$3'/" ./configs/retrieval_flickr.yaml;

python -m torch.distributed.launch --nproc_per_node=$1  --master_port=$(rand 2000 4000) train_retrieval.py \
--config ./configs/retrieval_flickr.yaml \
--output_dir output/retrieval_flickr_$2