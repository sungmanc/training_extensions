#!/bin/bash
DATA_ROOT=/home/sungmanc/datasets/classification

NUM_CLASSES=2
for DATA_NAME in "multiclass_CUB_small"
do
    for SUBSET in "1"
    do
        for MODEL in "otx_efficientnet_b0" "otx_mobilenet_v3_large" "otx_efficientnet_v2" "otx_deit_tiny"
        do
            python src/otx/cli/train.py \
            +recipe=classification/${MODEL} \
            model.otx_model.config.head.num_classes=${NUM_CLASSES} \
            model.otx_model.config.data_preprocessor.num_classes=${NUM_CLASSES} \
            trainer.max_epochs=20 \
            base.data_dir=${DATA_ROOT}/${DATA_NAME}/${SUBSET} \
            trainer=gpu
            # base.output_dir=outputs/${DATA_NAME}/${SUBSET}/${MODEL} \
        done
    done
done

# NUM_CLASSES=67
# for DATA_NAME in "multiclass_CUB_medium"
# do
#     for MODEL in "otx_efficientnet_b0" "otx_mobilenet_v3_large" "otx_efficientnet_v2" "otx_deit_tiny"
#     do
#         python src/otx/cli/train.py \
#         +recipe=classification/${MODEL} \
#         model.otx_model.config.head.num_classes=${NUM_CLASSES} \
#         model.otx_model.config.data_preprocessor.num_classes=${NUM_CLASSES} \
#         trainer.max_epochs=20 \
#         base.data_dir=${DATA_ROOT}/${DATA_NAME} \
#         trainer=gpu \
#         base.output_dir=outputs/${DATA_NAME}/${MODEL} \
#         hydra.job_logging.handlers.file.filename=outputs/outputs/${DATA_NAME}/${SUBSET}/${MODEL}
#     done
# done