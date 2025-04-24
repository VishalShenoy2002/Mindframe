import importlib.resources
import yaml
import os
import importlib

def load_config_yaml(filename:str):
    try:
        with importlib.resources.files("neuroforge.config").joinpath(filename).open("r") as f:
            data = yaml.safe_load(f)
            f.close()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{filename} not found in config dir.")
 
    return data