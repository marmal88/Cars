import yaml

CONFIG_LOCATION = "./src/config/config.yml"


def read_yaml_file():
    """Load yaml file"""
    with open(CONFIG_LOCATION, "r") as configf:
        config = yaml.load(configf, Loader=yaml.FullLoader)
    return config