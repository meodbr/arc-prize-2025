import os

from arc_tartiflette.utils import constants

def convert_env_var(value, var_type):
    if var_type == bool:
        return value.lower() in ("true", "1", "yes")
    elif var_type == list[str]:
        return value.split(",")
    else:
        return var_type(value)

def get_env_vars_with_defaults():
    env_vars = constants.DEFAULT_ENV_VARS
    returned_vars = {}
    for var, default in env_vars.items():
        returned_vars[var] = os.environ.get(var, default)["value"]
        returned_vars[var] = convert_env_var(returned_vars[var], default["type"])
    return returned_vars

ENV_VARS = get_env_vars_with_defaults()