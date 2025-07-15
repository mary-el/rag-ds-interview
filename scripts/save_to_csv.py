import argparse
from pathlib import Path

import pandas as pd

from configs import load_config


def save_to_csv(config_file: str, file_name: str):
    """
    Parse and save data to .csv
    """
    config = load_config(config_file)
    doc_files = [
        Path(config["doc_parsing"]["input_dir"]) / doc
        for doc in config["doc_parsing"]["files"]
    ]

    from scripts.doc_parser import parse_document

    dfs = []
    for doc_path in doc_files:
        dfs.append(parse_document(doc_path))

    df = pd.concat(dfs)
    df.to_csv(file_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="set config file",
        type=str,
        default="configs/config.yaml",
    )
    parser.add_argument(
        "--file",
        "-f",
        help="csv file name",
        type=str,
        default="data/df.csv",
    )

    args = parser.parse_args()
    save_to_csv(args.config, args.file)
