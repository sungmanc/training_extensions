from ote_cli.datasets import get_dataset_class
from ote_cli.registry import Registry
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.io import generate_label_schema, read_label_schema, read_model

from ote_sdk.configuration.helper import create

from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment

templates_dir = "external"
registry = Registry(templates_dir)
registry = registry.filter(task_type="detection")
model_template = registry.get('Custom_Object_Detection_Gen3_ATSS_ResNet50')

hyper_parameters = model_template.hyper_parameters.data
hyper_parameters = create(hyper_parameters)

Dataset = get_dataset_class(model_template.task_type)

def infer(trial: int, name: str, i: int):
    dataset = Dataset(
        test_subset={
            "ann_file": "/home/sungmanc/datasets/ssl_exp_sc_dataset/{}/annotations/instances_test.json".format(name),
            "data_root": "/home/sungmanc/datasets/ssl_exp_sc_dataset/{}/images/test".format(name)},
    )
    labels_schema = LabelSchemaEntity.from_labels(dataset.get_labels())

    Task = get_impl_class(model_template.entrypoints.base)

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=labels_schema,
        model_template=model_template)

    environment.model = read_model(
        environment.get_model_configuration(), "./outputs/ote_default_{}_16_{}/ATSS_coco_1x/weights.pth".format(name, i), None
        #environment.get_model_configuration(), "./outputs/pixpro_warm_with_padding/trial_{}/pixpro_warm_{}_16_{}/weights.pth".format(trial, name, i), None
        #environment.get_model_configuration(), "./outputs/pixpro_warm_without_padding/trial_{}/pixpro_warm_{}_16_{}/weights.pth".format(trial, name, i), None
    )

    task = Task(task_environment=environment)

    validation_dataset = dataset.get_subset(Subset.TESTING)
    
    task._config.evaluation['iou_thr'] = [
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    prediction_results, metrics = task._infer_detector(
        task._model,
        task._config,
        validation_dataset,
        dump_features=True, eval=True)
    
    predicted_validation_dataset = validation_dataset.with_empty_annotations()
    task._add_predictions_to_dataset(prediction_results, predicted_validation_dataset, task.confidence_threshold)

    resultset = ResultSetEntity(
        model=environment.model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )

    f1_score = task.evaluate(resultset)

    return resultset.performance.score.value, metrics, task

def infer(name: str, i: int):
    dataset = Dataset(
        test_subset={
            "ann_file": "/home/sungmanc/datasets/ssl_exp_sc_dataset/{}/annotations/instances_test.json".format(name),
            "data_root": "/home/sungmanc/datasets/ssl_exp_sc_dataset/{}/images/test".format(name)},
    )
    labels_schema = LabelSchemaEntity.from_labels(dataset.get_labels())

    Task = get_impl_class(model_template.entrypoints.base)

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=labels_schema,
        model_template=model_template)

    environment.model = read_model(
        environment.get_model_configuration(), "./outputs/ote_default_{}_16_{}/ATSS_coco_1x/weights.pth".format(name, i), None
    )

    task = Task(task_environment=environment)

    validation_dataset = dataset.get_subset(Subset.TESTING)
    
    task._config.evaluation['iou_thr'] = [
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    prediction_results, metrics = task._infer_detector(
        task._model,
        task._config,
        validation_dataset,
        dump_features=True, eval=True)
    
    predicted_validation_dataset = validation_dataset.with_empty_annotations()
    task._add_predictions_to_dataset(prediction_results, predicted_validation_dataset, task.confidence_threshold)

    resultset = ResultSetEntity(
        model=environment.model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )

    f1_score = task.evaluate(resultset)

    return resultset.performance.score.value, metrics, task


import pandas as pd

results = []
datasets = ['fish', 'vitens', 'pothole']
for name in datasets:
    for i in range(1, 6):
        f1, mAP, _ = infer(name, i)
        results += [(name, i, f1, mAP)]

df = pd.DataFrame(results, columns=["name", "subset", "F1", "mAP"])
df["mAP"] = 100.0 * df["mAP"]

print(df.groupby(["name"])["mAP"].mean())
df.to_csv('results_ote_default.csv')
#df.to_csv('results_pixpro_warm_with_padding.csv')
#df.to_csv('results_pixpro_warm_without_padding.csv')