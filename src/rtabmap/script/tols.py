
import json

def load_config(config_file):
    """Charge la configuration depuis un fichier JSON"""
    with open(config_file, 'r') as f:
        return json.load(f)

def config_to_args(config):
    """Convertit la configuration en arguments de ligne de commande pour RTABMap"""
    args = []
    
    # Les clés contiennent déjà le format complet "ExportCloudsDialog\parameter"
    for key, value in config.items():
        # Convertir les booléens en minuscules pour RTABMap
        if isinstance(value, bool):
            value_str = str(value).lower()
        else:
            value_str = str(value)
        
        arg = f"{key} {value_str}"
        args.append(arg)
    
    return args


