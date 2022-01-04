import os
import os.path as osp
import random as rd
import subprocess
import time
import numpy as np
from typing import Optional, Tuple, Union

import click

from mim.click import CustomCommand, param2lowercase
from mim.utils import (
    echo_success,
    exit_with_error,
    get_installed_path,
    highlighted_error,
    is_installed,
    module_full_name,
    recursively_find,
    get_usage
)


@click.command(
    name='mrun',
    context_settings=dict(ignore_unknown_options=True),
    cls=CustomCommand)
@click.argument('package', type=str, callback=param2lowercase)
@click.argument('config', type=str)
@click.option('-g', '--gpus', type=int, default=8, help='Number of gpus to use')
@click.option('-c', '--cpus', type=int, default=2, help='Number of cpus per task')
@click.option('-p', '--partition', type=str, default='mediaf', help='The partition to use')
def cli(package: str,
        config: str,
        gpus: int = 8,
        cpus: int = 2,
        partition: str = 'mediaf') -> None:

    is_success, msg = mrun(
        package=package,
        config=config,
        gpus=gpus,
        cpus=cpus,
        partition=partition)

    if is_success:
        echo_success(msg)  # type: ignore
    else:
        exit_with_error(msg)


def mrun(
    package: str,
    config: str,
    gpus: int = 8,
    cpus: int = 2,
    partition: str = 'mediaf',
) -> Tuple[bool, Union[str, Exception]]:

    full_name = module_full_name(package)
    if full_name == '':
        msg = f"Can't determine a unique package given abbreviation {package}"
        raise ValueError(highlighted_error(msg))
    package = full_name

    configs = [line for line in open(config).read().split('\n') if line != '']
    num_procs = len(configs)

    configs = [[x for x in line.split() if x != ''] for line in configs]
    for proc in configs:
        assert len(proc) > 0

    timestamp = time.strftime('%y%m%d_%H%M%S', time.localtime())
    bash_names = [f'mrun_{timestamp}_{i}.sh' for i in range(num_procs)]
    for i, proc in enumerate(configs):
        cmds = ['!/bin/bash']
        for cfg in proc:
            assert osp.exists(cfg), f"G! Config file {cfg} not exists! "
            port = rd.randint(20000, 30000)
            cmds.append(f'PORT={port} ./tools/dist_train.sh {cfg} {gpus} --validate --test-last --test-best')
        with open(bash_names[i], 'w') as fout:
            fout.write('\n'.join(cmds))

    for bash in bash_names:
        cmd = (
            f'srun -p {partition} --gres=gpu:{gpus} --ntasks=1 --ntasks-per-node=1 '
            f'--cpus-per-task={cpus * gpus} --kill-on-bad-exit=1 -x ./exclude.txt '
            f'bash {bash} &'
        )
        os.system(cmd)
    
    return True, 'Training finished successfully. '
