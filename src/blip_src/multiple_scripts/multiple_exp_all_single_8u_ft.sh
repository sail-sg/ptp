
declare -a PTMethodArray=( "Pretrain_concated_pred_4M" )

for pt_method in "${PTMethodArray[@]}"
do
    echo "==== start evaluate model $pt_method ===="

    echo "==== utilize pretrained model output/$pt_method/checkpoint_19.pth ===="

    bash ./multiple_scripts/exp_all_ft_single.sh $pt_method; 
done