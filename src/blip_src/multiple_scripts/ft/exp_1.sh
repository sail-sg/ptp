# image to text retrieval

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}


# ============================== step1: zero-shot retrieval evaluation ========= 
# evaluate on test 
echo '(coco step1: zero-shot evaluation) load pretrained model from: '$3;
sed -i "/^\(pretrained: \).*/s//\1'$3'/" ./configs/retrieval_coco.yaml;

python -m torch.distributed.launch --nproc_per_node=$1  --master_port=$(rand 2000 4000) train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco_$2 \
--evaluate

# print val than test set

# ================= step2: train and evaluate val set =================
echo '(coco step2: train on retrieval) load pretrained model from: '$3;
sed -i "/^\(pretrained: \).*/s//\1'$3'/" ./configs/retrieval_coco.yaml;

python -m torch.distributed.launch --nproc_per_node=$1  --master_port=$(rand 2000 4000) train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco_$2

# # ===========================step3: evaluate on val/test split ========= 
# # evaluate on test 

# TRAINED_MODEL="output\/retrieval_coco_${2}\/checkpoint_best.pth"
# echo '(coco step3: test/val eval) load trained retrieval model from: '${TRAINED_MODEL};

# sed -i "/^\(pretrained: \).*/s//\1'$TRAINED_MODEL'/" ./configs/retrieval_coco.yaml;


# python -m torch.distributed.launch --nproc_per_node=$1  --master_port=$(rand 2000 4000) train_retrieval.py \
# --config ./configs/retrieval_coco.yaml \
# --output_dir output/retrieval_coco_$2 \
# --evaluate