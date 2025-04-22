import yaml
import os

def load_config_yaml(filename:str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found. Please correct the path or create the file.")
    with open(filename,'r') as f:
        data = yaml.safe_load(f)
        f.close()
    
    return data