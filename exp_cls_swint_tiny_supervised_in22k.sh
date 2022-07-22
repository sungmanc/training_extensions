lr=0.007

data="cifar100"
for trial in "1"
do
    ote train \
    ./external/model-preparation-algorithm/configs/classification/swin_transformer_tiny_cls_incr/template.yaml \
    --train-ann-files=/home/sungmanc/datasets/${data}/train \
    --train-data-roots=/home/sungmanc/datasets/${data}/train \
    --val-ann-files=/home/sungmanc/datasets/${data}/test \
    --val-data-roots=/home/sungmanc/datasets/${data}/test \
    --load-weights=/home/sungmanc/scripts/pretrained_weights/supervised_swin_tiny_22k.pth \
    --save-model-to=./outputs/large_upm_cls_benchmark/full_data_finetune/cls_per_img_6/supervised_swin_tiny_22k_lr${lr}_${data}_${trial}/results
done

data="flower"
for trial in "1"
do
    ote train \
    ./external/model-preparation-algorithm/configs/classification/swin_transformer_tiny_cls_incr/template.yaml \
    --train-ann-files=/home/sungmanc/datasets/${data}/train \
    --train-data-roots=/home/sungmanc/datasets/${data}/train \
    --val-ann-files=/home/sungmanc/datasets/${data}/test \
    --val-data-roots=/home/sungmanc/datasets/${data}/test \
    --load-weights=/home/sungmanc/scripts/pretrained_weights/supervised_swin_tiny_22k.pth \
    --save-model-to=./outputs/large_upm_cls_benchmark/full_data_finetune/cls_per_img_6/supervised_swin_tiny_22k_lr${lr}_${data}_${trial}/results
done

data="lgchem"
for trial in "1"
do
    ote train \
    ./external/model-preparation-algorithm/configs/classification/swin_transformer_tiny_cls_incr/template.yaml \
    --train-ann-files=/home/sungmanc/datasets/${data}/train \
    --train-data-roots=/home/sungmanc/datasets/${data}/train \
    --val-ann-files=/home/sungmanc/datasets/${data}/test \
    --val-data-roots=/home/sungmanc/datasets/${data}/test \
    --load-weights=/home/sungmanc/scripts/pretrained_weights/supervised_swin_tiny_22k.pth \
    --save-model-to=./outputs/large_upm_cls_benchmark/full_data_finetune/cls_per_img_6/supervised_swin_tiny_22k_lr${lr}_${data}_${trial}/results
done

data="xray"
for trial in "1"
do
    ote train \
    ./external/model-preparation-algorithm/configs/classification/swin_transformer_tiny_cls_incr/template.yaml \
    --train-ann-files=/home/sungmanc/datasets/${data}/train \
    --train-data-roots=/home/sungmanc/datasets/${data}/train \
    --val-ann-files=/home/sungmanc/datasets/${data}/test \
    --val-data-roots=/home/sungmanc/datasets/${data}/test \
    --load-weights=/home/sungmanc/scripts/pretrained_weights/supervised_swin_tiny_22k.pth \
    --save-model-to=./outputs/large_upm_cls_benchmark/full_data_finetune/cls_per_img_6/supervised_swin_tiny_22k_lr${lr}_${data}_${trial}/results
done

