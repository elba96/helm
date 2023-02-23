from argparse import ArgumentParser
import json
from experiment import Experiment
from torch.multiprocessing import set_start_method
from multiprocessing import Process
import wandb
import yaml
import os
from functools import partial

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--run-id', type=str, help='Optional ID by which the current run should be saved')
    parser.add_argument('--config', type=str, metavar='CONFIG', help='Path to the config file',
                        default='config.json')
    parser.add_argument('--var', type=str, metavar='KEY=VALUE', action='append',
                        help='Key-value assignment for configuration variable - '
                             'will be updated in the current config file')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for Reproducibility')
    return parser.parse_args()


def parse_variable_assignment(assignments):
    vars = {}
    for ass in assignments:
        key, value = ass.split('=', 1)
        if "true" == value.lower() or "false" == value.lower():
            value = value.lower() == "true"
        elif 'e' in value or '.' in value:
            try:
                value = float(value)
            except ValueError:
                pass
        elif value.isdigit():
            value = int(value)
        vars[key] = value
    return vars


def run(config=None, run_id=None, seed=None):
    if config["wandb_log"]:
        wandb_run = wandb.init(
            sync_tensorboard=True,
            allow_val_change=True,
            monitor_gym=config.get("monitor_gym", False),
        )
        config = dict(wandb.config)
        run_id = f"{wandb_run.sweep_id}-{wandb_run.id}"
        seed = config["seed"]

    exp = Experiment(config, experiment_id=run_id)
    exp.run(seed=seed)


def wandb_run(config):
    def load_sweep_config(yaml_config_path):
        with open(yaml_config_path, "r") as f:
            try:
                sweep_config = yaml.safe_load(f)
                return sweep_config
            except yaml.YAMLError as err:
                raise err

    sweep_config = load_sweep_config(os.path.join("configs", "sweep-config.yaml"))

    for k, v in config.items():
        if k not in sweep_config["parameters"]:
            sweep_config["parameters"][k] = {"value": v}

    sweep_id = config.get("sweep_id", None)
    if not sweep_id:
        sweep_id = wandb.sweep(
            sweep_config,
        )
    wandb_func = partial(run, config)
    wandb.agent(sweep_id, function=wandb_func)


def main():
    options = create_parser()
    set_start_method('spawn')

    # load corresponding config
    config = json.load(open(options.config))

    if options.var is not None:
        updates = parse_variable_assignment(options.var)
        false_keys = [key for key in updates.keys() if key not in config]
        if len(false_keys):
            exc = ', '.join(false_keys)
            print(f"Added keys: {exc} to config...")
        config.update(updates)
        run_id = '_'.join([f'{k}={v}' for k, v in updates.items()])
    else:
        run_id = None

    if config.get("wandb_log", False):
        os.environ['WANDB_API_KEY'] = config.get("wandb_api_key")
        os.environ['WANDB_ENTITY'] = config.get("wandb_entity")
        os.environ['WANDB_PROJECT'] = config.get("wandb_project")
        del config["wandb_api_key"]
        del config["wandb_entity"]
        del config["wandb_project"]
        p = Process(target=wandb_run, args=(config,))
    else:
        p = Process(target=run, args=(config, run_id, options.seed))
    p.start()
    p.join()


if __name__ == '__main__':
    main()

