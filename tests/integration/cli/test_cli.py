# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
from otx.cli import main

# This assumes have OTX installed in environment.
otx_module = importlib.import_module("otx")
RECIPE_PATH = Path(inspect.getfile(otx_module)).parent / "recipe"
RECIPE_LIST = [str(p) for p in RECIPE_PATH.glob("**/*.yaml") if "_base_" not in p.parts]
RECIPE_OV_LIST = [str(p) for p in RECIPE_PATH.glob("**/openvino_model.yaml") if "_base_" not in p.parts]
RECIPE_LIST = set(RECIPE_LIST) - set(RECIPE_OV_LIST)

CLS_RECIPE_LIST = [str(p) for p in RECIPE_PATH.glob("classification/**/*.yaml")]
DET_RECIPE_LIST = [str(p) for p in RECIPE_PATH.glob("detection/**/*.yaml")]
ISEG_RECIPE_LIST = [str(p) for p in RECIPE_PATH.glob("instance_segmentation/**/*.yaml")]
RDET_RECIPE_LIST = [str(p) for p in RECIPE_PATH.glob("rotated_detection/**/*.yaml")]
SEG_RECIPE_LIST = [str(p) for p in RECIPE_PATH.glob("semantic_segmentation/**/*.yaml")]
VPM_RECIPE_LIST = [str(p) for p in RECIPE_PATH.glob("visual_prompting/**/*.yaml")]
ZSVP_RECIPE_LIST = [str(p) for p in RECIPE_PATH.glob("zero_shot_visual_prompting/**/*.yaml")]

def parametrize_by_task(task_name):
    recipe_lists = {
        'classification': CLS_RECIPE_LIST,
        'detection': DET_RECIPE_LIST,
        'instance_segmentation': ISEG_RECIPE_LIST,
        'rotated_detection': RDET_RECIPE_LIST,
        'semantic_segmentation': SEG_RECIPE_LIST,
        'visual_prompting': VPM_RECIPE_LIST,
        'zero_shot_visual_prompting': ZSVP_RECIPE_LIST,
    }
    def decorator(func):
        return pytest.mark.parametrize("recipe", recipe_lists[task_name])(func)
    return decorator

# Example usage
@parametrize_by_task('classification')
def test_classification(recipe):
    # Your test logic here
    pass

@parametrize_by_task('detection')
def test_detection(recipe):
    # Your test logic here
    pass


@pytest.mark.parametrize("recipe", RECIPE_LIST)
def test_otx_e2e(
    recipe: str,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
) -> None:
    """
    Test OTX CLI e2e commands.

    - 'otx train' with 2 epochs training
    - 'otx test' with output checkpoint from 'otx train'
    - 'otx export' with output checkpoint from 'otx train'
    - 'otx test' with the exported to ONNX/IR model

    Args:
        recipe (str): The recipe to use for training. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]
    if task in ("action_classification"):
        pytest.xfail(reason="xFail until this root cause is resolved on the Datumaro side.")

    # 1) otx train
    tmp_path_train = tmp_path / f"otx_train_{model_name}"
    command_cfg = [
        "otx",
        "train",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_train / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "2",
        *fxt_cli_override_command_per_task[task],
    ]

    with patch("sys.argv", command_cfg):
        main()

    if task in ("zero_shot_visual_prompting"):
        pytest.skip("Full CLI test is not applicable to this task.")

    # Currently, a simple output check
    assert (tmp_path_train / "outputs").exists()
    assert (tmp_path_train / "outputs" / "configs.yaml").exists()
    assert (tmp_path_train / "outputs" / "csv").exists()
    assert (tmp_path_train / "outputs" / "checkpoints").exists()
    ckpt_files = list((tmp_path_train / "outputs" / "checkpoints").glob(pattern="epoch_*.ckpt"))
    assert len(ckpt_files) > 0

    # 2) otx test
    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    command_cfg = [
        "otx",
        "test",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        fxt_accelerator,
        *fxt_cli_override_command_per_task[task],
        "--checkpoint",
        str(ckpt_files[-1]),
    ]

    with patch("sys.argv", command_cfg):
        main()

    assert (tmp_path_test / "outputs").exists()
    assert (tmp_path_test / "outputs" / "csv").exists()

    # 3) otx export
    if any(
        task_name in recipe
        for task_name in [
            "h_label_cls",
            "detection",
            "dino_v2",
            "instance_segmentation",
            "action",
            "visual_prompting",
        ]
    ):
        return

    format_to_file = {
        "ONNX": "exported_model.onnx",
        "OPENVINO": "exported_model.xml",
        "EXPORTABLE_CODE": "exportable_code.zip",
    }

    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    for fmt in format_to_file:
        command_cfg = [
            "otx",
            "export",
            "--config",
            recipe,
            "--data_root",
            fxt_target_dataset_per_task[task],
            "--engine.work_dir",
            str(tmp_path_test / "outputs"),
            *fxt_cli_override_command_per_task[task],
            "--checkpoint",
            str(ckpt_files[-1]),
            "--export_format",
            f"{fmt}",
        ]

        with patch("sys.argv", command_cfg):
            main()

        assert (tmp_path_test / "outputs").exists()
        assert (tmp_path_test / "outputs" / f"{format_to_file[fmt]}").exists()

    # 4) infer of the exported models
    task = recipe.split("/")[-2]
    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    if "_cls" in recipe:
        export_test_recipe = f"src/otx/recipe/classification/{task}/openvino_model.yaml"
    else:
        export_test_recipe = f"src/otx/recipe/{task}/openvino_model.yaml"
    exported_model_path = str(tmp_path_test / "outputs" / "exported_model.xml")

    command_cfg = [
        "otx",
        "test",
        "--config",
        export_test_recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        "cpu",
        *fxt_cli_override_command_per_task[task],
        "--model.model_name",
        exported_model_path,
    ]

    with patch("sys.argv", command_cfg):
        main()

    assert (tmp_path_test / "outputs").exists()


@pytest.mark.parametrize("recipe", RECIPE_LIST)
def test_otx_explain_e2e(
    recipe: str,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
) -> None:
    """
    Test OTX CLI explain e2e command.

    Args:
        recipe (str): The recipe to use for training. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    import cv2
    import numpy as np

    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]

    if ("_cls" not in task) and (task != "detection"):
        pytest.skip("Supported only for classification and detection task.")

    if "dino" in model_name:
        pytest.skip("Dino is not supported.")

    # otx explain
    tmp_path_explain = tmp_path / f"otx_explain_{model_name}"
    command_cfg = [
        "otx",
        "explain",
        "--config",
        recipe,
        "--model.num_classes",
        "1000",
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_explain / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--seed",
        "0",
        "--deterministic",
        "True",
        *fxt_cli_override_command_per_task[task],
    ]

    with patch("sys.argv", command_cfg):
        main()

    assert (tmp_path_explain / "outputs").exists()
    assert (tmp_path_explain / "outputs" / "saliency_map.tiff").exists()
    sal_map = cv2.imread(str(tmp_path_explain / "outputs" / "saliency_map.tiff"))
    assert sal_map.shape[0] > 0
    assert sal_map.shape[1] > 0

    reference_sal_vals = {
        "multi_label_cls_efficientnet_v2_light": np.array([66, 97, 84, 33, 42, 79, 0], dtype=np.uint8),
        "h_label_cls_efficientnet_v2_light": np.array([43, 84, 61, 5, 54, 31, 57], dtype=np.uint8),
    }
    test_case_name = task + "_" + model_name
    if test_case_name in reference_sal_vals:
        actual_sal_vals = sal_map[:, 0, 0]
        ref_sal_vals = reference_sal_vals[test_case_name]
        assert np.max(np.abs(actual_sal_vals - ref_sal_vals) <= 3)


@pytest.mark.parametrize("recipe", RECIPE_OV_LIST)
def test_otx_ov_test(recipe: str, tmp_path: Path, fxt_target_dataset_per_task: dict) -> None:
    """
    Test OTX CLI e2e commands.

    - 'otx test' with OV model

    Args:
        recipe (str): The OV recipe to use for testing. (eg. 'classification/openvino_model.yaml')
        tmp_path (Path): The temporary path for storing the testing outputs.

    Returns:
        None
    """
    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]

    if task in ["multi_label_cls", "instance_segmentation", "h_label_cls"]:
        # OMZ doesn't have proper model for Pytorch MaskRCNN interface
        # TODO(Kirill):  Need to change this test when export enabled #noqa: TD003
        pytest.skip("OMZ doesn't have proper model for these types of tasks.")

    # otx test
    tmp_path_test = tmp_path / f"otx_test_{task}_{model_name}"
    command_cfg = [
        "otx",
        "test",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        "cpu",
        "--disable-infer-num-classes",
    ]

    with patch("sys.argv", command_cfg):
        main()

    assert (tmp_path_test / "outputs").exists()
    assert (tmp_path_test / "outputs" / "csv").exists()
    metric_result = list((tmp_path_test / "outputs" / "csv").glob(pattern="**/metrics.csv"))
    assert len(metric_result) > 0
