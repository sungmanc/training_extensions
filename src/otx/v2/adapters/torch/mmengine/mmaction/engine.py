"""OTX adapters.torch.mmengine.mmaction.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mmaction.registry import VISUALIZERS

from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.mmaction.registry import MMActionRegistry
from otx.v2.adapters.torch.mmengine.mmdeploy import AVAILABLE
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.utils.logger import get_logger

if TYPE_CHECKING:
    import numpy as np


logger = get_logger()

MMACTION_DEFAULT_CONFIG_PATH = "src/otx/v2/configs/action_classification/otx_mmaction_classification_default.yaml"
MMACTION_DEFAULT_CONFIG = Config.fromfile(MMACTION_DEFAULT_CONFIG_PATH)

class MMActionEngine(MMXEngine):
    """The MMActionEngine class is responsible for running inference on pre-trained models."""

    def __init__(self, work_dir: str | Path | None = None) -> None:
        """Initialize a new instance of the MMActionEngine class.

        Args:
            work_dir (Optional[Union[str, Path]], optional): The working directory for the engine. Defaults to None.
            config (Optional[Union[Dict, Config, str]], optional): The configuration for the engine. Defaults to None.
        """
        super().__init__(work_dir=work_dir)
        self.registry = MMActionRegistry()

    def _update_eval_config(self, evaluator_config: dict | list[dict] | None) -> dict | list[dict] | None:
        if evaluator_config is None or not evaluator_config:
            evaluator_config = {
                "type": "AccMetric",
                "metric_list": ("top_k_accuracy", "mean_class_accuracy"),
            }
        return evaluator_config

    def _update_config(self, func_args: dict, **kwargs) -> tuple[Config, bool]:
        config, update_check = super()._update_config(func_args, **kwargs)

        for subset in ("val", "test"):
            if f"{subset}_dataloader" in config and config[f"{subset}_dataloader"] is not None:
                evaluator_config = self._get_value_from_config(f"{subset}_evaluator", func_args)
                config[f"{subset}_evaluator"] = self._update_eval_config(evaluator_config=evaluator_config)

        if hasattr(config, "visualizer") and config.visualizer.type not in VISUALIZERS:
            config.visualizer = {
                "type": "ActionVisualizer",
                "vis_backends": [{"type": "LocalVisBackend"}, {"type": "TensorboardVisBackend"}],
            }

        if hasattr(config, "param_scheduler"):
            config.param_scheduler = [
                {"type": "LinearLR", "by_epoch": True, "begin": 0, "end": 3},
                {"type": "CosineAnnealingLR"},
            ]
        return config, update_check

    def predict(
        self,
        model: torch.nn.Module | (dict | str) | None = None,
        img: str | (np.ndarray | list) | None = None,
        checkpoint: str | Path | None = None,
        pipeline: dict | list | None = None,
        device: str | (torch.device | None) = None,
        task: str | None = None,
        batch_size: int = 1,
        **kwargs,
    ) -> list[dict]:
        """Runs inference on the given input image(s) using the specified model and checkpoint.

        Args:
            model (Optional[Union[torch.nn.Module, Dict, str]], optional): The model to use for inference. Can be a
                PyTorch module, a dictionary containing the model configuration, or a string representing the path to
                the model checkpoint file. Defaults to None.
            img (Optional[Union[str, np.ndarray, list]], optional): The input image(s) to run inference on. Can be a
                string representing the path to the image file, a NumPy array containing the image data, or a list of
                NumPy arrays containing multiple images. Defaults to None.
            checkpoint (Optional[Union[str, Path]], optional): The path to the checkpoint file to use for inference.
                Defaults to None.
            pipeline (Optional[Union[Dict, List]], optional): The data pipeline to use for inference. Can be a
                dictionary containing the pipeline configuration, or a list of dictionaries containing multiple
                pipeline configurations. Defaults to None.
            device (Union[str, torch.device, None], optional): The device to use for inference. Can be a string
                representing the device name (e.g. 'cpu' or 'cuda'), a PyTorch device object, or None to use the
                default device. Defaults to None.
            task (Optional[str], optional): The type of task to perform. Defaults to None.
            batch_size (int, optional): The batch size to use for inference. Defaults to 1.
            **kwargs: Additional keyword arguments to pass to the inference function.

        Returns:
            List[Dict]: A list of dictionaries containing the inference results.
        """
        from mmaction.apis.inferencers import MMAction2Inferencer

        # Model config need data_pipeline of test_dataloader
        # Update pipelines
        if pipeline is None:
            from otx.v2.adapters.torch.mmengine.mmaction.dataset import get_default_pipeline
            pipeline = get_default_pipeline(subset="predict")
        config = Config({})
        if isinstance(model, torch.nn.Module) and hasattr(model, "_config"):
            config = model._config  # noqa: SLF001
        elif isinstance(model, dict) and "_config" in model:
            config = model["_config"]
        config["test_dataloader"] = {"dataset": {"pipeline": pipeline}}
        if isinstance(model, dict):
            model.setdefault("_config", config)
        elif isinstance(model, torch.nn.Module):
            model._config = config  # noqa: SLF001

        if isinstance(checkpoint, Path):
            checkpoint = str(checkpoint)
        if task is not None and task != "VideoRecognition":
            raise NotImplementedError

        inferencer = MMAction2Inferencer(
            rec=config,
            rec_weights=checkpoint,
            device=device,
            input_format="rawframes",
        )

        return inferencer(img, batch_size=batch_size, **kwargs)

    def export(
        self,
        model: torch.nn.Module | (str | Config) | None = None,
        checkpoint: str | Path | None = None,
        precision: str | None = "float32",  # ["float16", "fp16", "float32", "fp32"]
        task: str | None = "VideoRecognition",
        codebase: str | None = "mmaction",
        export_type: str = "OPENVINO",  # "ONNX" or "OPENVINO"
        deploy_config: str | dict | None = None,
        device: str = "cpu",
        input_shape: tuple[int, int] | None = None,
    ) -> dict:
        """Export a PyTorch model to a specified format for deployment.

        Args:
            model (Optional[Union[torch.nn.Module, str, Config]]): The PyTorch model to export.
            checkpoint (Optional[Union[str, Path]]): The path to the checkpoint file to use for exporting.
            precision (Optional[str]): The precision to use for exporting.
                Can be one of ["float16", "fp16", "float32", "fp32"].
            task (Optional[str]): The task for which the model is being exported. Defaults to "VideoRecognition".
            codebase (Optional[str]): The codebase for the model being exported. Defaults to "mmaction".
            export_type (str): The type of export to perform. Can be one of "ONNX" or "OPENVINO". Defaults to "OPENVINO"
            deploy_config (Optional[str]): The path to the deployment configuration file to use for exporting.
                File path only.
            device (str): The device to use for exporting. Defaults to "cpu".
            input_shape (Optional[Tuple[int, int]]): The input shape of the model being exported.
            **kwargs: Additional keyword arguments to pass to the export function.

        Returns:
            dict: A dictionary containing information about the exported model.
        """
        if not AVAILABLE:
            msg = "MMXEngine's export is dependent on mmdeploy."
            raise ModuleNotFoundError(msg)
        from mmdeploy.utils import get_backend_config, get_codebase_config, get_ir_config, load_config

        from otx.v2.adapters.torch.mmengine.mmdeploy.exporter import Exporter
        from otx.v2.adapters.torch.mmengine.mmdeploy.utils.deploy_cfg_utils import (
            patch_input_preprocessing,
            patch_input_shape,
        )

        # Configure model_cfg
        model_cfg = None
        if model is not None:
            if isinstance(model, str):
                model_cfg = Config.fromfile(model)
            elif isinstance(model, Config):
                model_cfg = copy.deepcopy(model)
            elif isinstance(model, torch.nn.Module) and hasattr(model, "_config"):
                model_cfg = model._config.get("model", model._config)  # noqa: SLF001
            else:
                raise NotImplementedError
        elif self.dumped_config.get("model", None) and self.dumped_config["model"] is not None:
            if isinstance(self.dumped_config["model"], dict):
                model_cfg = Config(self.dumped_config["model"])
            else:
                model_cfg = self.dumped_config["model"]
        else:
            msg = "Not fount target model."
            raise ValueError(msg)

        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)
        self.dumped_config["model"] = model_cfg
        self.dumped_config["default_scope"] = "mmengine"

        # deploy_cfg
        codebase_config = None
        ir_config = None
        backend_config = None
        if deploy_config is not None:
            deploy_config_dict = load_config(deploy_config)[0]
            ir_config = get_ir_config(deploy_config_dict)
            backend_config = get_backend_config(deploy_config_dict)
            codebase_config = get_codebase_config(deploy_config_dict)
        else:
            deploy_config_dict = {}

        # codebase_config update
        if codebase_config is None:
            codebase = codebase if codebase is not None else self.registry.name
            codebase_config = {"type": codebase, "task": task}
            deploy_config_dict["codebase_config"] = codebase_config

        # if_config update
        if ir_config is None:
            ir_config = {
                "type": "onnx",
                "export_params": True,
                "keep_initializers_as_inputs": False,
                "opset_version": 11,
                "save_file": "end2end.onnx",
                "input_names": ["input"],
                "output_names": ["output"],
                "input_shape": None,
                "optimize": False,
                "dynamic_axes": {
                    "input": {
                        0: "batch",
                        1: "num_crops * num_segs",
                        3: "time",
                        4: "height",
                        5: "width",
                    },
                    "output": {
                        0: "batch",
                    },
                },
            }
            deploy_config_dict["ir_config"] = ir_config
        # BACKEND_CONFIG Update
        if backend_config is None:
            backend_config = {
                "type": "openvino",
                "model_inputs": [
                    {
                        "opt_shapes":{
                            "input": [1, 1, 3, 32, 224, 224],
                        },
                    },
                ],
                "mo_options":{
                    "args":{
                        "--source_layout": "?bctwh",
                    },
                },
            }
            deploy_config_dict["backend_config"] = backend_config

        # Patch input's configuration
        if isinstance(deploy_config_dict, dict):
            deploy_config_dict = Config(deploy_config_dict)
        data_preprocessor = self.dumped_config.model.get("data_preprocessor", None)
        mean = data_preprocessor["mean"] if data_preprocessor is not None else [123.675, 116.28, 103.53]
        std = data_preprocessor["std"] if data_preprocessor is not None else [58.395, 57.12, 57.375]
        to_rgb = data_preprocessor["to_rgb"] if data_preprocessor is not None else False
        patch_input_preprocessing(deploy_cfg=deploy_config_dict, mean=mean, std=std, to_rgb=to_rgb)
        if not deploy_config_dict.backend_config.get("model_inputs", []):
            if input_shape is None:
                pass
            patch_input_shape(deploy_config_dict, input_shape=input_shape)

        export_dir = Path(self.work_dir) / f"{self.timestamp}_export"
        exporter = Exporter(
            config=self.dumped_config,
            checkpoint=str(checkpoint),
            deploy_config=deploy_config_dict,
            work_dir=str(export_dir),
            precision=precision,
            export_type=export_type,
            device=device,
        )

        return exporter.export()