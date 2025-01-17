# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import pytest

from tests.regression.summarize_test_results import summarize_results_data


@pytest.fixture(autouse=True, scope="session")
def run_regression_tests(tmp_dir_path):
    print(f"tmp dir path = {tmp_dir_path}")
    yield

    input_path = os.path.join(tmp_dir_path, "regression_test_results")
    output_path = os.environ.get("TOX_WORK_DIR", os.getcwd())

    summarize_results_data(input_path, output_path)
