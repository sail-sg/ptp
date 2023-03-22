for VARIABLE in {0..11..1}
do
    mkdir CC3M/$VARIABLE
    ./azcopy copy https://biglmdiag.blob.core.windows.net/vinvl/image_features/googlecc_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/$VARIABLE/predictions.tsv \
    CC3M/$VARIABLE --recursive
    ./azcopy copy https://biglmdiag.blob.core.windows.net/vinvl/image_features/googlecc_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/$VARIABLE/predictions.lineidx \
    CC3M/$VARIABLE --recursive
    ./azcopy copy https://biglmdiag.blob.core.windows.net/vinvl/image_features/googlecc_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/$VARIABLE/annotations \
    CC3M/$VARIABLE --recursive
done