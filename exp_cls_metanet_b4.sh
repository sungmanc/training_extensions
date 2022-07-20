data="cifar100"

for trial in "1"
do
    ote train \
    ./external/model-preparation-algorithm/configs/classification/metanet_b4_cls_incr/template.yaml \
    --train-ann-files=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
    --train-data-roots=/home/sungmanc/datasets/${data}_cls_per_img_6/train \
    --val-ann-files=/home/sungmanc/datasets/${data}/test \
    --val-data-roots=/home/sungmanc/datasets/${data}/test \
    --save-model-to=./outputs/large_upm_cls_benchmark/cls_per_img_6/metanet_b4_${data}_${trial}/results \
    --enable-hpo
done