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
"""A CLI to create or update pipelines."""
from __future__ import absolute_import

import os

import argparse
import json
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():

    # Parses command-line arguments.
    from pipelines import _utils as utils
    args = utils.get_parse_args().parse_args()

    # Executes the appropriate pipeline module based on the provided command-line argument.
    if (args.class_name == "create_pipeline"):
        from pipelines import create_pipeline
        create_pipeline.main()
    elif (args.class_name == "run_pipeline"):
        from pipelines import run_pipeline
        run_pipeline.main() 
    else:
        print("No Class was selected")

if __name__ == '__main__':
	main()