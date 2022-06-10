# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import defaultdict

import pytest
from detection_tasks.apis.detection.config_utils import (
    cluster_anchors,
    config_from_string,
    config_to_string,
    get_data_cfg,
    is_epoch_based_runner,
    patch_adaptive_repeat_dataset,
    patch_config,
    patch_datasets,
    prepare_for_testing,
    prepare_for_training,
    prepare_work_dir,
    remove_from_config,
    save_config_to_file,
    set_data_classes,
    set_hyperparams,
)
from detection_tasks.apis.detection.configuration import OTEDetectionConfig
from mmcv import Config
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback


class TestConfigUtilsInputParamsValidation:
    @e2e_pytest_unit
    def test_is_epoch_based_runner_input_params_validation(self):
        """
        <b>Description:</b>
        Check "is_epoch_based_runner" function input parameters validation

        <b>Input data:</b>
        "runner_config" non-ConfigDict object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "runner_config" function
        """
        with pytest.raises(ValueError):
            is_epoch_based_runner(runner_config="unexpected_str")  # type: ignore

    @e2e_pytest_unit
    def test_patch_config_input_params_validation(self):
        """
        <b>Description:</b>
        Check "patch_config" function input parameters validation

        <b>Input data:</b>
        "patch_config" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "patch_config" function
        """
        label = LabelEntity(name="test label", domain=Domain.SEGMENTATION)
        correct_values_dict = {
            "config": Config(),
            "work_dir": "./work_dir",
            "labels": [label],
            "domain": Domain.DETECTION
        }
        unexpected_float = 1.1
        unexpected_values = [
            # Unexpected float is specified as "config" parameter
            ("config", unexpected_float),
            # Unexpected float is specified as "work_dir" parameter
            ("work_dir", unexpected_float),
            # Empty string is specified as "work_dir" parameter
            ("work_dir", ""),
            # String with null-character is specified as "work_dir" parameter
            ("work_dir", "null\0character/path"),
            # String with non-printable character is specified as "work_dir" parameter
            ("work_dir", "\non_printable_character/path"),
            # Unexpected float is specified as "labels" parameter
            ("labels", unexpected_float),
            # Unexpected float is specified as nested "label"
            ("labels", [label, unexpected_float]),
            # Unexpected float is specified as "domain" parameter
            ("domain", unexpected_float),
            # Unexpected float is specified as "random_seed" parameter
            ("random_seed", unexpected_float),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=patch_config,
        )

    @e2e_pytest_unit
    def test_set_hyperparams_input_params_validation(self):
        """
        <b>Description:</b>
        Check "set_hyperparams" function input parameters validation

        <b>Input data:</b>
        "set_hyperparams" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "set_hyperparams" function
        """
        correct_values_dict = {
            "config": Config(),
            "hyperparams": OTEDetectionConfig(header="config header"),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "config" parameter
            ("config", unexpected_str),
            # Unexpected string is specified as "hyperparams" parameter
            ("hyperparams", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=set_hyperparams,
        )

    @e2e_pytest_unit
    def test_patch_adaptive_repeat_dataset_input_params_validation(self):
        """
        <b>Description:</b>
        Check "patch_adaptive_repeat_dataset" function input parameters validation

        <b>Input data:</b>
        "patch_adaptive_repeat_dataset" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "patch_adaptive_repeat_dataset" function
        """
        correct_values_dict = {"config": Config(), "num_samples": 10}
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "config" parameter
            ("config", unexpected_str),
            # Unexpected string is specified as "num_samples" parameter
            ("num_samples", unexpected_str),
            # Unexpected string is specified as "decay" parameter
            ("decay", unexpected_str),
            # Unexpected string is specified as "factor" parameter
            ("factor", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=patch_adaptive_repeat_dataset,
        )

    @e2e_pytest_unit
    def test_prepare_for_testing_input_params_validation(self):
        """
        <b>Description:</b>
        Check "prepare_for_testing" function input parameters validation

        <b>Input data:</b>
        "prepare_for_testing" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "prepare_for_testing" function
        """
        correct_values_dict = {"config": Config(), "dataset": DatasetEntity()}
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "config" parameter
            ("config", unexpected_str),
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=prepare_for_testing,
        )

    @e2e_pytest_unit
    def test_prepare_for_training_input_params_validation(self):
        """
        <b>Description:</b>
        Check "prepare_for_training" function input parameters validation

        <b>Input data:</b>
        "prepare_for_training" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "prepare_for_training" function
        """
        dataset = DatasetEntity()
        time_monitor = TimeMonitorCallback(
            num_epoch=5, num_train_steps=2, num_val_steps=1, num_test_steps=1
        )
        correct_values_dict = {
            "config": Config(),
            "train_dataset": dataset,
            "val_dataset": dataset,
            "time_monitor": time_monitor,
            "learning_curves": defaultdict(),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "config" parameter
            ("config", unexpected_str),
            # Unexpected string is specified as "train_dataset" parameter
            ("train_dataset", unexpected_str),
            # Unexpected string is specified as "val_dataset" parameter
            ("val_dataset", unexpected_str),
            # Unexpected string is specified as "time_monitor" parameter
            ("time_monitor", unexpected_str),
            # Unexpected string is specified as "learning_curves" parameter
            ("learning_curves", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=prepare_for_training,
        )

    @e2e_pytest_unit
    def test_config_to_string_input_params_validation(self):
        """
        <b>Description:</b>
        Check "config_to_string" function input parameters validation

        <b>Input data:</b>
        "config" non-Config type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "config_to_string" function
        """
        with pytest.raises(ValueError):
            config_to_string(config=1)  # type: ignore

    @e2e_pytest_unit
    def test_config_from_string_input_params_validation(self):
        """
        <b>Description:</b>
        Check "config_from_string" function input parameters validation

        <b>Input data:</b>
        "config_string" non-string type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "config_from_string" function
        """
        with pytest.raises(ValueError):
            config_from_string(config_string=1)  # type: ignore

    @e2e_pytest_unit
    def test_save_config_to_file_input_params_validation(self):
        """
        <b>Description:</b>
        Check "save_config_to_file" function input parameters validation

        <b>Input data:</b>
        "config" non-Config type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "save_config_to_file" function
        """
        with pytest.raises(ValueError):
            save_config_to_file(config=1)  # type: ignore

    @e2e_pytest_unit
    def test_prepare_work_dir_input_params_validation(self):
        """
        <b>Description:</b>
        Check "prepare_work_dir" function input parameters validation

        <b>Input data:</b>
        "config" non-Config type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "prepare_work_dir" function
        """
        with pytest.raises(ValueError):
            prepare_work_dir(config=1)  # type: ignore

    @e2e_pytest_unit
    def test_set_data_classes_input_params_validation(self):
        """
        <b>Description:</b>
        Check "set_data_classes" function input parameters validation

        <b>Input data:</b>
        "set_data_classes" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "set_data_classes" function
        """
        label = LabelEntity(name="test label", domain=Domain.SEGMENTATION)
        correct_values_dict = {"config": Config(), "labels": [label]}
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "config" parameter
            ("config", unexpected_str),
            # Unexpected string is specified as "dataset" parameter
            ("labels", unexpected_str),
            # Unexpected string is specified as nested "label"
            ("labels", [label, unexpected_str]),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=set_data_classes,
        )

    @e2e_pytest_unit
    def test_patch_datasets_input_params_validation(self):
        """
        <b>Description:</b>
        Check "patch_datasets" function input parameters validation

        <b>Input data:</b>
        "config" non-Config type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "patch_datasets" function
        """
        correct_values_dict = {"config": Config(), "domain": Domain.DETECTION}
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "config" parameter
            ("config", unexpected_str),
            # Unexpected string is specified as "domain" parameter
            ("domain", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=patch_datasets,
        )

    @e2e_pytest_unit
    def test_remove_from_config_input_params_validation(self):
        """
        <b>Description:</b>
        Check "remove_from_config" function input parameters validation

        <b>Input data:</b>
        "remove_from_config" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "remove_from_config" function
        """
        correct_values_dict = {"config": Config(), "key": "key_1"}
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "config" parameter
            ("config", unexpected_int),
            # Unexpected integer is specified as "key" parameter
            ("key", unexpected_int),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=remove_from_config,
        )

    @e2e_pytest_unit
    def test_cluster_anchors_input_params_validation(self):
        """
        <b>Description:</b>
        Check "cluster_anchors" function input parameters validation

        <b>Input data:</b>
        "cluster_anchors" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "cluster_anchors" function
        """
        correct_values_dict = {
            "config": Config(),
            "dataset": DatasetEntity(),
            "model": None,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "config" parameter
            ("config", unexpected_str),
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
            # Unexpected string is specified as nested "label"
            ("model", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=cluster_anchors,
        )

    @e2e_pytest_unit
    def test_get_data_cfg_input_params_validation(self):
        """
        <b>Description:</b>
        Check "get_data_cfg" function input parameters validation

        <b>Input data:</b>
        "get_data_cfg" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "get_data_cfg" function
        """
        correct_values_dict = {
            "config": Config(),
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "config" parameter
            ("config", unexpected_int),
            # Unexpected integer is specified as "subset" parameter
            ("subset", unexpected_int),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_data_cfg,
        )
