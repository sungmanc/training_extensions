import json
import os

from tests.test_suite.run_test_command import (
    get_template_dir,
    check_run,
)


def regression_eval_testing(
    template,
    root,
    otx_dir,
    args,
    criteria,
    result_dict,
    threshold=0.10,
):
    regression_result = {
        "passed": True,
        "log": "",
    }

    template_work_dir = get_template_dir(template, root)

    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth",
        "--output",
        f"{template_work_dir}/trained_{template.model_template_id}",
    ]
    command_line.extend(["--workspace", f"{template_work_dir}"])
    command_line.extend(args.get("eval_params", []))
    check_run(command_line)

    performance_json_path = f"{template_work_dir}/trained_{template.model_template_id}/performance.json"
    assert os.path.exists(performance_json_path)

    with open(performance_json_path) as read_file:
        trained_performance = json.load(read_file)

    model_criteria = criteria[template.name] if template.name in criteria.keys() else 0.0
    modified_criteria = model_criteria - (model_criteria * threshold)
    for k in trained_performance.keys():
        result_dict[k] = round(trained_performance[k], 3)
        if trained_performance[k] < modified_criteria:
            regression_result["passed"] = False
            regression_result["log"] = f"Performance: ({trained_performance[k]}) < Criteria: ({modified_criteria})."
            regression_result["raw"] = {
                "metric": k,
                "performance": trained_performance[k],
                "template": template.name,
                "criteria": model_criteria,
                "threshold": threshold,
            }

    result_dict["Model size (MB)"] = round(
        os.path.getsize(f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth") / 1e6, 2
    )

    return regression_result


def regression_openvino_testing(
    template,
    root,
    otx_dir,
    args,
    threshold=0.0,
    criteria=None,
    reg_threshold=0.10,
    result_dict=None,
    half_precision=False,
):
    regression_result = {
        "passed": True,
        "log": "",
    }

    template_work_dir = get_template_dir(template, root)
    weights_path = f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml"
    output_path = f"{template_work_dir}/exported_{template.model_template_id}"
    perf_path = f"{template_work_dir}/exported_{template.model_template_id}/performance.json"

    if half_precision:
        weights_path = f"{template_work_dir}/exported_{template.model_template_id}_fp16/openvino.xml"
        output_path = f"{template_work_dir}/exported_{template.model_template_id}_fp16"
        perf_path = f"{template_work_dir}/exported_{template.model_template_id}_fp16/performance.json"

    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        weights_path,
        "--output",
        output_path,
    ]
    command_line.extend(["--workspace", f"{template_work_dir}"])
    check_run(command_line)

    trained_perf_path = f"{template_work_dir}/trained_{template.model_template_id}/performance.json"
    assert os.path.exists(trained_perf_path)
    with open(trained_perf_path) as read_file:
        trained_performance = json.load(read_file)

    assert os.path.exists(perf_path)
    with open(perf_path) as read_file:
        exported_performance = json.load(read_file)

    model_criteria = 0.0  # set default model critera for not existing reg config
    if template.name not in criteria.keys():
        regression_result["passed"] = False
        log_msg = (
            f"Cannot find regression criteria for the template '{template.name}'. "
            + f"train_performance = {trained_performance}, export_performance = {exported_performance}"
        )
        regression_result["log"] = log_msg
        print(log_msg)
        return regression_result

    if isinstance(criteria, dict):
        model_criteria = criteria[template.name] * (1.0 - reg_threshold)

    for k in trained_performance.keys():
        result_dict[k] = round(exported_performance[k], 3)
        if exported_performance[k] < model_criteria:
            regression_result["passed"] = False
            regression_result[
                "log"
            ] = f"Export performance: ({exported_performance[k]}) < Criteria: ({model_criteria})."

        if (
            exported_performance[k] < trained_performance[k]
            and abs(trained_performance[k] - exported_performance[k]) / (trained_performance[k] + 1e-10) > threshold
        ):
            regression_result["passed"] = False
            regression_result["log"] = f"{trained_performance[k]=}, {exported_performance[k]=}"

    return regression_result


def regression_deployment_testing(
    template, root, otx_dir, args, threshold=0.0, criteria=None, reg_threshold=0.10, result_dict=None
):
    regression_result = {
        "passed": True,
        "log": "",
    }

    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/deployed_{template.model_template_id}/openvino.zip",
        "--output",
        f"{template_work_dir}/deployed_{template.model_template_id}",
    ]
    command_line.extend(["--workspace", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/deployed_{template.model_template_id}/performance.json")
    with open(f"{template_work_dir}/exported_{template.model_template_id}/performance.json") as read_file:
        exported_performance = json.load(read_file)
    with open(f"{template_work_dir}/deployed_{template.model_template_id}/performance.json") as read_file:
        deployed_performance = json.load(read_file)

    if isinstance(criteria, dict) and template.name in criteria.keys():
        model_criteria = criteria[template.name]
        modified_criteria = model_criteria - (model_criteria * reg_threshold)

    for k in exported_performance.keys():
        if isinstance(criteria, dict) and template.name in criteria.keys():
            result_dict[k] = round(deployed_performance[k], 3)
            if deployed_performance[k] < modified_criteria:
                regression_result["passed"] = False
                regression_result[
                    "log"
                ] = f"Deploy performance: ({deployed_performance[k]}) < Criteria: ({modified_criteria})."
        if (
            deployed_performance[k] < exported_performance[k]
            and abs(exported_performance[k] - deployed_performance[k]) / (exported_performance[k] + 1e-10) > threshold
        ):
            regression_result["passed"] = False
            regression_result["log"] = f"{exported_performance[k]=}, {deployed_performance[k]=}"

    return regression_result


def regression_nncf_eval_testing(
    template, root, otx_dir, args, threshold=0.01, criteria=None, reg_threshold=0.10, result_dict=None
):
    regression_result = {
        "passed": True,
        "log": "",
    }

    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/nncf_{template.model_template_id}/weights.pth",
        "--output",
        f"{template_work_dir}/nncf_{template.model_template_id}",
    ]
    command_line.extend(["--workspace", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/nncf_{template.model_template_id}/performance.json")
    with open(f"{template_work_dir}/nncf_{template.model_template_id}/nncf_performance.json") as read_file:
        trained_performance = json.load(read_file)
    with open(f"{template_work_dir}/nncf_{template.model_template_id}/performance.json") as read_file:
        evaluated_performance = json.load(read_file)

    if isinstance(criteria, dict) and template.name in criteria.keys():
        model_criteria = criteria[template.name]
        modified_criteria = model_criteria - (model_criteria * reg_threshold)

    for k in trained_performance.keys():
        if isinstance(criteria, dict) and template.name in criteria.keys():
            result_dict[k] = round(evaluated_performance[k], 3)
            if evaluated_performance[k] < modified_criteria:
                regression_result["passed"] = False
                regression_result[
                    "log"
                ] = f"NNCF performance: ({evaluated_performance[k]}) < Criteria: ({modified_criteria})."
        if (
            evaluated_performance[k] < trained_performance[k]
            and abs(trained_performance[k] - evaluated_performance[k]) / (trained_performance[k] + 1e-10) > threshold
        ):
            regression_result["passed"] = False
            regression_result["log"] = f"{trained_performance[k]=}, {evaluated_performance[k]=}"

    return regression_result


def regression_ptq_eval_testing(template, root, otx_dir, args, criteria=None, reg_threshold=0.10, result_dict=None):
    regression_result = {
        "passed": True,
        "log": "",
    }

    template_work_dir = get_template_dir(template, root)
    command_line = [
        "otx",
        "eval",
        template.model_template_path,
        "--test-data-roots",
        f'{os.path.join(otx_dir, args["--test-data-roots"])}',
        "--load-weights",
        f"{template_work_dir}/ptq_{template.model_template_id}/openvino.xml",
        "--output",
        f"{template_work_dir}/ptq_{template.model_template_id}",
    ]
    command_line.extend(["--workspace", f"{template_work_dir}"])
    check_run(command_line)
    assert os.path.exists(f"{template_work_dir}/ptq_{template.model_template_id}/performance.json")

    with open(f"{template_work_dir}/ptq_{template.model_template_id}/performance.json") as read_file:
        ptq_performance = json.load(read_file)

    if isinstance(criteria, dict) and template.name in criteria.keys():
        model_criteria = criteria[template.name]
        modified_criteria = model_criteria - (model_criteria * reg_threshold)

    for k in ptq_performance.keys():
        if isinstance(criteria, dict) and template.name in criteria.keys():
            result_dict[k] = round(ptq_performance[k], 3)
            if ptq_performance[k] < modified_criteria:
                regression_result["passed"] = False
                regression_result["log"] = f"POT performance: ({ptq_performance[k]}) < Criteria: ({modified_criteria})."

    return regression_result


def regression_train_time_testing(train_time_criteria, e2e_train_time, template, threshold=0.30):
    """Measure train+val time and comapre with test criteria.

    Test criteria was set by previous measurement.
    """
    regression_result = {
        "passed": True,
        "log": "",
    }

    e2e_train_time_criteria = train_time_criteria[template.name] if template.name in train_time_criteria.keys() else 0.0
    modified_train_criteria = e2e_train_time_criteria + (e2e_train_time_criteria * threshold)

    if e2e_train_time > modified_train_criteria:
        regression_result["passed"] = False
        regression_result["log"] = f"Train time: ({e2e_train_time}) < Criteria: ({modified_train_criteria})."

    return regression_result


def regression_eval_time_testing(eval_time_criteria, e2e_eval_time, template, threshold=0.30):
    """Measure evaluation time and comapre with test criteria.

    Test criteria was set by previous measurement.
    """
    regression_result = {
        "passed": True,
        "log": "",
    }

    e2e_eval_time_criteria = eval_time_criteria[template.name] if template.name in eval_time_criteria.keys() else 0.0
    modified_eval_criteria = e2e_eval_time_criteria + (e2e_eval_time_criteria * threshold)

    if e2e_eval_time > modified_eval_criteria:
        regression_result["passed"] = False
        regression_result["log"] = f"Eval time: ({e2e_eval_time}) < criteria: ({modified_eval_criteria})."

    return regression_result
