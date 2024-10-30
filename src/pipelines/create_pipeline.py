"""A CLI to create or update pipelines."""
from __future__ import absolute_import

import os

import argparse
import json
import sys

from pipelines._utils import get_pipeline_driver, convert_struct

def main():  # pragma: no cover
    """The main harness that creates or updates and runs the pipeline.

    Creates or updates the pipeline and runs it.
    """

    # Parses command-line arguments.
    from pipelines import _utils as utils
    args = utils.get_parse_args().parse_args()

    if args.module_name is None or args.role_arn is None:
        #parser.print_help()
        utils.get_parse_args().print_help()
        sys.exit(2)

    tags = convert_struct(args.tags)

    print("\n")
    print("Args: " + str(args))
    print("Know Args: " + args.kwargs)
    print("\n")
   
    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        print("###### Creating/updating a SageMaker Pipeline with the following definition:")
        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        upsert_response = pipeline.upsert(
            role_arn=args.role_arn, description=args.description, tags=tags
        )
        print("\n###### Created/Updated SageMaker Pipeline: Response received:")
        print(upsert_response)

    except Exception as e: 
        print(f"Exception: {e}")
        print("")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)


if __name__ == "__main__":
    main()