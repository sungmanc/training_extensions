for lr in "0.0007" "0.007" "0.07" "0.7" "1.0"
do
    data="cifar100"
    for trial in "2" "3"
    do
        ote train \
        ./external/model-preparation-algorithm/configs/classification/swin_transformer_tiny_cls_incr/template.yaml \
        --train-ann-files=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
        --train-data-roots=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
        --val-ann-files=/home/sungmanc/datasets/${data}/test \
        --val-data-roots=/home/sungmanc/datasets/${data}/test \
        --load-weights=/home/sungmanc/scripts/pretrained_weights/unicl_swin_tiny_in21k_yfcc14m.pth \
        --save-model-to=./outputs/large_upm_cls_benchmark/few_shot_linear_probe/cls_per_img_6/unicl_swin_tiny_in21k_yfcc14m_lr${lr}_${data}_${trial}/results \
        params --learning_parameters.learning_rate ${lr} --learning_parameters.num_iters 20
    done

    data="flower"
    for trial in "2" "3"
    do
        ote train \
        ./external/model-preparation-algorithm/configs/classification/swin_transformer_tiny_cls_incr/template.yaml \
        --train-ann-files=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
        --train-data-roots=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
        --val-ann-files=/home/sungmanc/datasets/${data}/test \
        --val-data-roots=/home/sungmanc/datasets/${data}/test \
        --load-weights=/home/sungmanc/scripts/pretrained_weights/unicl_swin_tiny_in21k_yfcc14m.pth \
        --save-model-to=./outputs/large_upm_cls_benchmark/few_shot_linear_probe/cls_per_img_6/unicl_swin_tiny_in21k_yfcc14m_lr${lr}_${data}_${trial}/results \
        params --learning_parameters.learning_rate ${lr} --learning_parameters.num_iters 20        
    done

    data="lgchem"
    for trial in "2" "3"
    do
        ote train \
        ./external/model-preparation-algorithm/configs/classification/swin_transformer_tiny_cls_incr/template.yaml \
        --train-ann-files=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
        --train-data-roots=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
        --val-ann-files=/home/sungmanc/datasets/${data}/test \
        --val-data-roots=/home/sungmanc/datasets/${data}/test \
        --load-weights=/home/sungmanc/scripts/pretrained_weights/unicl_swin_tiny_in21k_yfcc14m.pth \
        --save-model-to=./outputs/large_upm_cls_benchmark/few_shot_linear_probe/cls_per_img_6/unicl_swin_tiny_in21k_yfcc14m_lr${lr}_${data}_${trial}/results \
        params --learning_parameters.learning_rate ${lr} --learning_parameters.num_iters 20
    done

    data="xray"
    for trial in "2" "3"
    do
        ote train \
        ./external/model-preparation-algorithm/configs/classification/swin_transformer_tiny_cls_incr/template.yaml \
        --train-ann-files=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
        --train-data-roots=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
        --val-ann-files=/home/sungmanc/datasets/${data}/test \
        --val-data-roots=/home/sungmanc/datasets/${data}/test \
        --load-weights=/home/sungmanc/scripts/pretrained_weights/unicl_swin_tiny_in21k_yfcc14m.pth \
        --save-model-to=./outputs/large_upm_cls_benchmark/few_shot_linear_probe/cls_per_img_6/unicl_swin_tiny_in21k_yfcc14m_lr${lr}_${data}_${trial}/results \
        params --learning_parameters.learning_rate ${lr} --learning_parameters.num_iters 20
    done
done
