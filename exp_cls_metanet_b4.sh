for lr in "0.0007" "0.007" "0.07" "0.1" "1.0"
do
    data="cifar100"
    for trial in "1"
    do
        ote train \
        ./external/model-preparation-algorithm/configs/classification/metanet_b4_cls_incr/template.yaml \
        --train-ann-files=/local/sungmanc/datasets/${data}_cls_per_img_6/train \
        --train-data-roots=/local/sungmanc/datasets/${data}_cls_per_img_6/train \
        --val-ann-files=/local/sungmanc/datasets/${data}/test \
        --val-data-roots=/local/sungmanc/datasets/${data}/test \
        --save-model-to=./outputs/large_upm_cls_benchmark/few_shot_finetune/cls_per_img_6/metanet_b4_lr${lr}_${data}_${trial}/results \
        params --learning_parameters.learning_rate ${lr}
    done

    data="flower"
    for trial in "1"
    do
        ote train \
        ./external/model-preparation-algorithm/configs/classification/metanet_b4_cls_incr/template.yaml \
        --train-ann-files=/local/sungmanc/datasets/${data}_cls_per_img_6/train \
        --train-data-roots=/local/sungmanc/datasets/${data}_cls_per_img_6/train \
        --val-ann-files=/local/sungmanc/datasets/${data}/test \
        --val-data-roots=/local/sungmanc/datasets/${data}/test \
        --save-model-to=./outputs/large_upm_cls_benchmark/few_shot_finetune/cls_per_img_6/metanet_b4_lr${lr}_${data}_${trial}/results \
        params --learning_parameters.learning_rate ${lr}
    done

    data="lgchem"
    for trial in "1"
    do
        ote train \
        ./external/model-preparation-algorithm/configs/classification/metanet_b4_cls_incr/template.yaml \
        --train-ann-files=/local/sungmanc/datasets/${data}_cls_per_img_6/train \
        --train-data-roots=/local/sungmanc/datasets/${data}_cls_per_img_6/train \
        --val-ann-files=/local/sungmanc/datasets/${data}/test \
        --val-data-roots=/local/sungmanc/datasets/${data}/test \
        --save-model-to=./outputs/large_upm_cls_benchmark/few_shot_finetune/cls_per_img_6/metanet_b4_lr${lr}_${data}_${trial}/results \
        params --learning_parameters.learning_rate ${lr}
    done

    data="xray"
    for trial in "1"
    do
        ote train \
        ./external/model-preparation-algorithm/configs/classification/metanet_b4_cls_incr/template.yaml \
        --train-ann-files=/local/sungmanc/datasets/${data}_cls_per_img_6/train \
        --train-data-roots=/local/sungmanc/datasets/${data}_cls_per_img_6/train \
        --val-ann-files=/local/sungmanc/datasets/${data}/test \
        --val-data-roots=/local/sungmanc/datasets/${data}/test \
        --save-model-to=./outputs/large_upm_cls_benchmark/few_shot_finetune/cls_per_img_6/metanet_b4_lr${lr}_${data}_${trial}/results \
        params --learning_parameters.learning_rate ${lr}
    done

done




