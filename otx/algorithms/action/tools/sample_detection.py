"""Sample Code of otx training for action detection."""

# Copyright (C) 2021-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import sys

from mmcv.utils import get_logger

from otx.algorithms.common.utils import get_task_class
from otx.api.configuration.helper import create
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.core.data.adapter import get_dataset_adapter

logger = get_logger(name="sample")


def parse_args():
    """Parse function for getting model template & check export."""
    parser = argparse.ArgumentParser(description="Sample showcasing the new API")
    parser.add_argument("template_file_path", help="path to template file")
    parser.add_argument("--export", action="store_true")
    return parser.parse_args()


TRAIN_DATA_ROOTS = "data/cvat_dataset/action_detection/train"
VAL_DATA_ROOTS = "data/cvat_dataset/action_detection/train"


def load_test_dataset(model_template):
    """Load Sample dataset for detection."""
    dataset_adapter = get_dataset_adapter(
        model_template.task_type,
        train_data_roots=TRAIN_DATA_ROOTS,
        val_data_roots=VAL_DATA_ROOTS,
    )
    dataset = dataset_adapter.get_otx_dataset()
    label_schema = dataset_adapter.get_label_schema()
    return dataset, label_schema


# pylint: disable=too-many-locals, too-many-statements
def main(args):
    """Main function of Detection Sample."""
    logger.info("Fine tuning sample dataset")
    logger.info("Sample dataset can be found at data/custom_action_recognition/custom_dataset")

    logger.info("Load model template")
    model_template = parse_model_template(args.template_file_path)

    logger.info("Get dataset")
    dataset, labels_schema = load_test_dataset(model_template)

    logger.info("Set hyperparameters")
    params = create(model_template.hyper_parameters.data)
    params.learning_parameters.num_iters = 5

    logger.info("Setup environment")
    environment = TaskEnvironment(
        model=None,
        hyper_parameters=params,
        label_schema=labels_schema,
        model_template=model_template,
    )

    logger.info("Create base Task")
    task_impl_path = model_template.entrypoints.base
    task_cls = get_task_class(task_impl_path)
    task = task_cls(task_environment=environment)

    logger.info("Train model")
    output_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
    )
    task.train(dataset, output_model)

    logger.info("Get predictions on the validation set")
    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=False),
    )
    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    logger.info("Estimate quality on validation set")
    task.evaluate(resultset)
    logger.info(str(resultset.performance))

    if args.export:
        raise Exception("CVS-102941 ONNX export of action detection model keeps failed")


if __name__ == "__main__":
    sys.exit(main(parse_args()) or 0)