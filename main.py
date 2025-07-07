import argparse

from configs import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sync", "-s", help="synchronize db with docs", action="store_true"
    )
    parser.add_argument(
        "--config",
        "-c",
        help="set config file",
        type=str,
        default="configs/config.yaml",
    )
    parser.add_argument("--bot", "-b", help="run telegram bot", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)  # get config from the file

    if args.bot:  # start Telegram bot
        from bot.handlers import run_bot

        print(f"Telegram bot: https://t.me/{config['interface']['bot_name']}")
        run_bot()
    else:
        from app.loop import run_loop  # run main loop

        run_loop()
