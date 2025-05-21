import argparse

from configs import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sync", "-s", help="synchronize db with docs", action="store_true"
    )
    parser.add_argument('--config', '-c', help="config file", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    load_config(args.config)  # get config from the file

    from app.loop import run_loop
    run_loop()
