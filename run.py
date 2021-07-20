"""
The main file for the Multi-omic integration comparison project.
The arguments entered into the Command-Line Interface (CLI) are parsed here,
and based on those arguments the required functionalities are called.
A .yml from the /configs folder is the preferred setup.

Adding functionalities is as easy as adding a small parser using the argparse library,
and creating a function here that calls the functionalities. We've tried to keep as much of the
logic outside this file, only setup functions are required.

Authors:
    Stavros Makrodimitris    S.Makrodimitris@tudelft.nl
    Bram Pronk               I.B.Pronk@student.tudelft.nl
"""
import sys

import yaml
import argparse

from src.MVAE.train import run as run_MVAE

# Argument Parser
PARSER = argparse.ArgumentParser(prog='run.py', description="Generic runner for joint data integration models")
PARSER.add_argument('--config', '-c',
                    dest='config_file',
                    metavar='FILE',
                    help="path to the config file",
                    default='configs/main.yaml')
PARSER.add_argument('-poe',
                    action='store_true',
                    help="Running the Product-of-Experts MVAE")


def main() -> None:
    """
    The start of the program. This function determines from user input which function handler it should call.

    @return: None
    """
    # Print all the user entered arguments in a neatly organized fashion
    args = PARSER.parse_args()

    if args.config_file is None:
        print("No config file specified. Use -c '/path/to/config.yaml' to insert one. Exiting program.")
        sys.exit()

    with open(args.config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Incorrect config.yaml file!")
            print(exc)

    if args.poe:
        print("Running Product-of-Experts MVAE Model")
        print({**config['GLOBAL_PARAMS'], **config['MVAE']})
        run_MVAE({**config['GLOBAL_PARAMS'], **config['MVAE']})


if __name__ == '__main__':
    main()
