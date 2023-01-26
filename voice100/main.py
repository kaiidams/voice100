# Copyright (C) 2023 Katsuya Iida. All rights reserved.

from pytorch_lightning.cli import LightningCLI
from voice100.models import Voice100ModelBase
from voice100.data_modules import Voice100DataModuleBase


def cli_main():
    cli = LightningCLI(  # noqa: F841
        Voice100ModelBase,
        Voice100DataModuleBase,
        subclass_mode_model=True,
        subclass_mode_data=True)


if __name__ == "__main__":
    cli_main()
