# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Provides utilities for SageMaker Pipeline CLI."""
from __future__ import absolute_import

import ast
import argparse

def get_pipeline_driver(module_name, passed_args=None):
    """Gets the driver for generating your pipeline definition.

    Pipeline modules must define a get_pipeline() module-level method.

    Args:
        module_name: The module name of your pipeline.
        passed_args: Optional passed arguments that your pipeline may be templated by.

    Returns:
        The SageMaker Workflow pipeline.
    """
    _imports = __import__(module_name, fromlist=["get_pipeline"])
    kwargs = convert_struct(passed_args)
    return _imports.get_pipeline(**kwargs)

def convert_struct(str_struct=None):
    return ast.literal_eval(str_struct) if str_struct else {}

def get_parse_args():
    
    #Defines command-line arguments to specify the pipeline module and optional keyword arguments.
    parser = argparse.ArgumentParser(
        "Creates or updates and runs the pipeline for the pipeline script."
    )

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="The module name of the pipeline to import.",
    )
    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )
    parser.add_argument(
        "-description",
        "--description",
        dest="description",
        type=str,
        default=None,
        help="The description of the pipeline.",
    )
    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )

    parser.add_argument(
        "-c",
        "--class-name",
        dest="class_name",
        type=str,
        help="The class name of the pipeline to execute.",
    )

    return parser
