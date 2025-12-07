import os
import json

from dotenv import load_dotenv

from arc_tartiflette.config import defaults


def convert_env_var(value, var_type):
    if var_type == bool:
        return value.lower() in ("true", "1", "yes")
    elif var_type == list[str]:
        return value.split(",")
    else:
        return var_type(value)


def get_env_vars_with_defaults():
    env_vars = defaults.DEFAULT_ENV_VARS
    returned_vars = {}
    for var, default in env_vars.items():
        returned_vars[var] = os.environ.get(var, default["value"])
        returned_vars[var] = convert_env_var(returned_vars[var], default["type"])
    return returned_vars


def refresh_env_vars():
    load_dotenv()
    return get_env_vars_with_defaults()


def get_logging_config():
    log_config_path = os.environ.get("LOGGING_CONFIG_PATH", "configs/logging.json")
    with open(log_config_path, "r") as f:
        return json.load(f)


ENV_VARS = refresh_env_vars()
