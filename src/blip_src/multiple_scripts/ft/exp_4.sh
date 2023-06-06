# NLVR2

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}

echo '(NLVR:) load pretrained model from: '$3;
sed -i "/^\(pretrained: \).*/s//\1'$3'/" ./configs/nlvr.yaml;

python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$(rand 2000 4000) train_nlvr.py \
--config ./configs/nlvr.yaml \
--output_dir output/NLVR_$2 # --evaluate