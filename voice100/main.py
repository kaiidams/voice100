# Copyright (C) 2023 Katsuya Iida. All rights reserved.

from pytorch_lightning.cli import LightningCLI
import voice100.models  # noqa: F401
import voice100.data_modules  # noqa: F401


def cli_main():
    cli = LightningCLI()


if __name__ == "__main__":
    cli_main()
