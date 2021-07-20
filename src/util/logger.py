"""
Functions for all output of the program.
Use instead of print() to save output to a log file.
"""
import os
import sys
import datetime as dt

# Global that is altered in main() in run.py, where the saving directory is setup
output_file = "SET_IN_RUN.PY"


def info(message: str) -> None:
    """
    This functions is used for all output that is used as info.
    This prints in white
    :param message: The info message to be printed
    :return: None
    """
    print(message)

    write_to_log(message)


def error(message: str) -> None:
    """
    This functions is used for all output that is used as error message.
    This prints in red
    :param message: The error message to be printed
    :return: None
    """
    if "win32" in sys.platform.lower():
        print("ERROR: " + message)
    else:
        print("\033[1;31mERROR: %s\033[0;0m" % message)

    write_to_log("ERROR: " + message)


def success(message: str) -> None:
    """
    This functions is used for all output that is used as success message.
    This prints in green.
    :param message: The success message to be printed
    :return: None
    """
    if "win32" in sys.platform.lower():
        print("SUCCESS: " + message)
    else:
        print("\033[1;32mSUCCESS: %s\033[0;0m" % message)

    write_to_log("SUCCESS: " + message)


def request_input(message: str) -> str:
    """
    This function is used for all input required for the program.
    :param message: The message to print before input
    :return: A string of the user's inout
    """
    if "win32" in sys.platform.lower():
        input_res = input(message)
    else:
        input_res = input("\033[0;0m%s" % message)

    write_to_log(("{0} {1}".format(message, input_res)))

    return input_res


def write_to_log(message: str) -> None:
    """
    This functions writes all output to a log file
    :param message:
    :return:
    """
    path = os.path.join(output_file, "output.txt")

    with open(path, "a") as file:
        file.write(str(dt.datetime.now().replace(microsecond=0)) + ": " + message + "\n")
        file.close()
