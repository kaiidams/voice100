# main.py
from pytorch_lightning.cli import LightningCLI
import voice100.models  # noqa: F401
import voice100.data_modules  # noqa: F401

def cli_main():
    cli = LightningCLI()
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block