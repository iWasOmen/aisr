import yaml
from pathlib import Path
import os
def get_project_root():
    current_path = Path(__file__)
    return current_path.parent.parent.parent
def load_conifg():
    config_name = "config.yaml"
    config_path = get_project_root()/config_name
    with open(config_path,'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config

config = load_conifg()