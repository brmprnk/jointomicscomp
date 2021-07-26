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
import os

import argparse
import yaml
from datetime import datetime

import src.util.logger as logger

# This is the Project Root, important that run.py is inside this folder for accurate ROOT_DIR
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Argument Parser
PARSER = argparse.ArgumentParser(prog='run.py', description="Generic runner for joint data integration models")
PARSER.add_argument('--config', '-c',
                    dest='config_file',
                    metavar='FILE',
                    help="path to the config file",
                    default='configs/main.yaml')
PARSER.add_argument('-mofadata',
                    action='store_true',
                    help="Preprocessing data for Multi-Omics Factor Analysis V2 (MOFA+)")
PARSER.add_argument('-mofa',
                    action='store_true',
                    help="Running Multi-Omics Factor Analysis V2 (MOFA+)")
PARSER.add_argument('-moe',
                    action='store_true',
                    help="Running Mixture-of-Experts MVAE")
PARSER.add_argument('-poe',
                    action='store_true',
                    help="Running Product-of-Experts MVAE")
PARSER.add_argument('-mvib',
                    action='store_true',
                    help="Running Multi-View Information Bottleneck")
PARSER.add_argument('-cgae',
                    action='store_true',
                    help="Running CGAE Model")


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

    # Create directory to store all results in
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    save_dir = os.path.join(ROOT_DIR, 'results', '{} {}'.format(config['GLOBAL_PARAMS']['name'], dt_string))
    os.makedirs(save_dir)
    config['GLOBAL_PARAMS']['save_dir'] = save_dir
    config['GLOBAL_PARAMS']['ROOT_DIR'] = ROOT_DIR

    # Setup save directory for output logging
    logger.output_file = save_dir

    if args.mofadata:
        from src.MOFA2.create_mofa_data import create_mofa_dataframe

        create_mofa_dataframe({**config['GLOBAL_PARAMS']})
        sys.exit()

    # If no specific model set, run all models and end program
    if not args.mofa and not args.moe and not args.poe and not args.mvib and not args.cgae:
        run_mofa(config)
        run_mvae(config, mixture=True, product=True)
        run_mvib(config)
        run_cgae(config)
        return

    # Run models individually / combined
    if args.mofa:
        run_mofa(config)

    if args.moe:
        run_mvae(config, mixture=True)

    if args.poe:
        run_mvae(config, product=True)

    if args.mvib:
        run_mvib(config)

    if args.cgae:
        run_cgae(config)

    return


def run_mofa(config: dict) -> None:
    """
    Setup and run MOFA+ module.

    @param config: Dictionary containing input parameters
    @return: None
    """
    from src.MOFA2.mofa import run as mofa2

    mofa2({**config['GLOBAL_PARAMS'], **config['MOFA+']})


def run_mvae(config: dict, mixture=False, product=False) -> None:
    """
    Setup and run MOFA+ module.

    @param config: Dictionary containing input parameters
    @param mixture: boolean flag that indicates whether the MVAE model is Mixture-of-Experts
    @param product: boolean flag that indicates whether the MVAE model is Product-of-Experts
    @return: None
    """
    from src.MVAE.train import run as mvae_model

    if mixture is True:
        print("Running Mixture-of-Experts MVAE Model")

        config['MVAE']['mixture'] = True
        mvae_model({**config['GLOBAL_PARAMS'], **config['MVAE']})
    if product is True:
        print("Running Product-of-Experts MVAE Model")

        config['MVAE']['mixture'] = False
        mvae_model({**config['GLOBAL_PARAMS'], **config['MVAE']})


def run_mvib(config: dict) -> None:
    """
    Setup and run MOFA+ module.

    @param config: Dictionary containing input parameters
    @return: None
    """
    print("Multi-view Information Bottleneck has not yet been implemented")


def run_cgae(config: dict) -> None:
    """
    Setup and run MOFA+ module.

    @param config: Dictionary containing input parameters
    @return: None
    """
    print("CGAE has not yet been implemented")


if __name__ == '__main__':
    main()
