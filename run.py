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
                    default='configs/gegcn.yaml')
PARSER.add_argument('--experiment', '-e',
                    help="Name of experiment",
                    default="experiment")
PARSER.add_argument('--results-path', '-R',
                    dest='results_path',
                    help="where to store the results",
                    default="/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp")
PARSER.add_argument('-survival',
                    action='store_true',
                    help="Run Task 2 for survival time comparison")
PARSER.add_argument('-baseline',
                    action='store_true',
                    help="Run the baseline and its evaluation")
PARSER.add_argument('-mofa',
                    action='store_true',
                    help="Running Multi-Omics Factor Analysis V2 (MOFA+)")
PARSER.add_argument('-moe',
                    action='store_true',
                    help="Running Mixture-of-Experts MVAE")
PARSER.add_argument('-poe',
                    action='store_true',
                    help="Running Product-of-Experts MVAE")
PARSER.add_argument('-mvae-impute',
                    action='store_true',
                    help="Calls the prediction module of the MVAE")
PARSER.add_argument('-mvib',
                    action='store_true',
                    help="Running Multi-View Information Bottleneck")
PARSER.add_argument('-cgae',
                    action='store_true',
                    help="Running CGAE Model")
PARSER.add_argument('-cvae',
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
    if args.experiment != "experiment":
        config['GLOBAL_PARAMS']['name'] = args.experiment
    save_dir = os.path.join(args.results_path, 'results', '{}_{}_{}'
                            .format(config['GLOBAL_PARAMS']['name'],
                                    config['GLOBAL_PARAMS']['data1'],
                                    config['GLOBAL_PARAMS']['data2']))

    os.makedirs(save_dir)
    config['GLOBAL_PARAMS']['save_dir'] = save_dir
    config['GLOBAL_PARAMS']['ROOT_DIR'] = ROOT_DIR

    # Setup save directory for output logging
    logger.output_file = save_dir
    logger.info("Running the JointOmicsComp software - A compilation of multi-omic integration models.")

    logger.success("Starting Experiment : {}".format(args.experiment))
    logger.success("Saving to : {}".format(save_dir))

    # Check for utility arguments, run only these and exit program
    if args.survival:
        from src.survival import run as survival_comparison
        survival_comparison({**config['GLOBAL_PARAMS']})
        sys.exit()

    # If no specific model set, run all models and end program
    if not args.baseline and not args.mofa and not args.moe and not args.poe and not args.mvib and not args.cgae \
            and not args.mvae_impute and not args.cvae:
        run_baseline(config)
        run_mofa(config)
        run_poe(config)
        run_moe(config)
        run_mvib(config)
        run_cgae(config)
        run_cvae(config)
        return

    # Run models individually / combined
    if args.baseline:
        run_baseline(config)

    if args.mofa:
        run_mofa(config)

    if args.moe:
        assert 'MoE' in config
        run_moe(config)

    if args.poe:
        assert 'PoE' in config
        run_poe(config)

    if args.mvib:
        run_mvib(config)

    if args.cgae:
        run_cgae(config)

    if args.cvae:
        run_cvae(config)

    # Run special function
    if args.mvae_impute:
        mvae_impute(config)


def run_baseline(config: dict) -> None:
    """
    Setup and run baseline

    @param config: Dictionary containing input parameters
    @return: None
    """
    import src.baseline.baseline as baseline

    logger.info("\n##########")

    baseline.run_baseline({**config['GLOBAL_PARAMS'], **config['BASELINE']})

    logger.info("##########\n")


def run_mofa(config: dict) -> None:
    """
    Setup and run MOFA+ module.

    @param config: Dictionary containing input parameters
    @return: None
    """
    from src.MOFA2.mofa import run as mofa2

    logger.info("\n##########")

    mofa2({**config['GLOBAL_PARAMS'], **config['MOFA+']})

    logger.info("##########\n")


def run_poe(config: dict) -> None:
    """
    Setup and run product of experts.

    @param config: Dictionary containing input parameters
    @return: None
    """
    from src.PoE.train import run as mvae_model

    logger.info("\n##########")
    logger.success("Running Product-of-Experts MVAE Model")


    mvae_model({**config['GLOBAL_PARAMS'], **config['PoE']})
    logger.success("Finished running Product-of-Experts MVAE Model")
    logger.info("##########\n")



def run_moe(config: dict) -> None:
    """
    Setup and run mixture of experts.

    @param config: Dictionary containing input parameters
    @return: None
    """
    from src.MoE.train import run as mvae_model

    logger.info("\n##########")
    logger.success("Running Mixture-of-Experts MVAE Model")


    mvae_model({**config['GLOBAL_PARAMS'], **config['MoE']})
    logger.success("Finished running Mixture-of-Experts MVAE Model")
    logger.info("##########\n")





def run_mvib(config: dict) -> None:
    """
    Setup and run multi-view info bottleneck.

    @param config: Dictionary containing input parameters
    @return: None
    """
    from src.MVIB.train_representation import run as mvib_model


    logger.info("\n##########")

    mvib_model({**config['GLOBAL_PARAMS'], **config['MVIB']})

    logger.info("##########\n")


def run_cgae(config: dict) -> None:
    """
    Setup and run CGVAE.

    @param config: Dictionary containing input parameters
    @return: None
    """
    from src.CGAE.main import run as cgae_model

    logger.info("\n##########")

    cgae_model({**config['GLOBAL_PARAMS'], **config['CGAE']})

    logger.info("##########\n")



def run_cvae(config: dict) -> None:
    """
    Setup and run CVAE.

    @param config: Dictionary containing input parameters
    @return: None
    """
    from src.CVAE.main import run as cvae_model

    logger.info("\n##########")

    cvae_model({**config['GLOBAL_PARAMS'], **config['CVAE']})

    logger.info("##########\n")




def mvae_impute(config: dict):
    from src.MVAE.evaluate import predict

    logger.info("\n##########")

    predict({**config['GLOBAL_PARAMS'], **config['MVAE']})

    logger.info("##########\n")


if __name__ == '__main__':
    main()
