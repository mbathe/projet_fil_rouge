
import json


def load_config(config_file):
    """Charge la configuration depuis un fichier JSON"""
    with open(config_file, 'r') as f:
        return json.load(f)


def config_to_args(config):
    """Convertit la configuration en arguments de ligne de commande pour RTABMap"""
    args = []
    for key, value in config.items():
        if isinstance(value, bool):
            value_str = str(value).lower()
        else:
            value_str = str(value)
        # Split key and value for subprocess.run
        args.extend([str(key), value_str])
    return args
