lr=0.007

data="cifar100"
for trial in "1"
do
    ote train \
    ./external/model-preparation-algorithm/configs/classification/swin_transformer_tiny_cls_incr/template.yaml \
    --train-ann-files=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
    --train-data-roots=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
    --val-ann-files=/home/sungmanc/datasets/${data}/test \
    --val-data-roots=/home/sungmanc/datasets/${data}/test \
    --load-weights=/home/sungmanc/scripts/pretrained_weights/unicl_swin_tiny_in21k_yfcc14m.pth \
    --save-model-to=./outputs/large_upm_cls_benchmark/few_shot_finetune_hpo/cls_per_img_6/unicl_swin_tiny_in21k_yfcc14m_${data}_${trial}/results \
    --enable-hpo
done

data="flower"
for trial in "1"
do
    ote train \
    ./external/model-preparation-algorithm/configs/classification/swin_transformer_tiny_cls_incr/template.yaml \
    --train-ann-files=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
    --train-data-roots=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
    --val-ann-files=/home/sungmanc/datasets/${data}/test \
    --val-data-roots=/home/sungmanc/datasets/${data}/test \
    --load-weights=/home/sungmanc/scripts/pretrained_weights/unicl_swin_tiny_in21k_yfcc14m.pth \
    --save-model-to=./outputs/large_upm_cls_benchmark/few_shot_finetune_hpo/cls_per_img_6/unicl_swin_tiny_in21k_yfcc14m_${data}_${trial}/results \
    --enable-hpo
done

data="lgchem"
for trial in "1"
do
    ote train \
    ./external/model-preparation-algorithm/configs/classification/swin_transformer_tiny_cls_incr/template.yaml \
    --train-ann-files=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
    --train-data-roots=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
    --val-ann-files=/home/sungmanc/datasets/${data}/test \
    --val-data-roots=/home/sungmanc/datasets/${data}/test \
    --load-weights=/home/sungmanc/scripts/pretrained_weights/unicl_swin_tiny_in21k_yfcc14m.pth \
    --save-model-to=./outputs/large_upm_cls_benchmark/few_shot_finetune_hpo/cls_per_img_6/unicl_swin_tiny_in21k_yfcc14m_${data}_${trial}/results \
    --enable-hpo
done

data="xray"
for trial in "1"
do
    ote train \
    ./external/model-preparation-algorithm/configs/classification/swin_transformer_tiny_cls_incr/template.yaml \
    --train-ann-files=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
    --train-data-roots=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
    --val-ann-files=/home/sungmanc/datasets/${data}/test \
    --val-data-roots=/home/sungmanc/datasets/${data}/test \
    --load-weights=/home/sungmanc/scripts/pretrained_weights/unicl_swin_tiny_in21k_yfcc14m.pth \
    --save-model-to=./outputs/large_upm_cls_benchmark/few_shot_finetune_hpo/cls_per_img_6/unicl_swin_tiny_in21k_yfcc14m_${data}_${trial}/results \
    --enable-hpo
done

