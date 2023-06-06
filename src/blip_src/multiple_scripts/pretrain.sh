python move_pretrained_weights.py;

# 4M \
python -m torch.distributed.launch --nproc_per_node=8 pretrain_concated_pred_tsv.py \
--config ./configs/pretrain_concated_pred_4M.yaml --output_dir output/Pretrain_concated_pred_4M

echo "output dir is: output/Pretrain_concated_pred_4M"