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

import argparse
import yaml

from src.MVAE.train import run as mvae_model

# Argument Parser
PARSER = argparse.ArgumentParser(prog='run.py', description="Generic runner for joint data integration models")
PARSER.add_argument('--config', '-c',
                    dest='config_file',
                    metavar='FILE',
                    help="path to the config file",
                    default='configs/main.yaml')
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

    # No specific model set, run all models and end program
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
    print("MOFA has not yet been implemented")


def run_mvae(config: dict, mixture=False, product=False) -> None:
    """
    Setup and run MOFA+ module.

    @param config: Dictionary containing input parameters
    @param mixture: boolean flag that indicates whether the MVAE model is Mixture-of-Experts
    @param product: boolean flag that indicates whether the MVAE model is Product-of-Experts
    @return: None
    """
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
