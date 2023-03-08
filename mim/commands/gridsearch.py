# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import itertools
import os
import os.path as osp
import random as rd
import subprocess
import sys
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor as Executor
from typing import Optional, Tuple, Union

import click

from mim.click import CustomCommand
from mim.utils import (
    args2string,
    echo_error,
    echo_success,
    exit_with_error,
    get_config,
    get_installed_path,
    highlighted_error,
    is_installed,
    module_full_name,
    recursively_find,
    set_config,
    string2args,
    get_usage
)

PYTHON = sys.executable


@click.command(
    name='gridsearch',
    context_settings=dict(ignore_unknown_options=True),
    cls=CustomCommand)
@click.argument('package', type=str)
@click.argument('config', type=str)
@click.option(
    '-l',
    '--launcher',
    type=click.Choice(['pytorch', 'slurm'], case_sensitive=False),
    default='pytorch',
    help='Job launcher')
@click.option(
    '--port',
    type=int,
    default=None,
    help=('The port used for inter-process communication (only applicable to '
          'slurm / pytorch launchers). If set to None, will randomly choose '
          'a port between 20000 and 30000. '))
@click.option(
    '-G', '--gpus', type=int, default=8, help='Number of gpus to use')
@click.option(
    '-g',
    '--gpus-per-node',
    type=int,
    default=8,
    help=('Number of gpus per node to use '
          '(only applicable to launcher == "slurm")'))
@click.option(
    '-c',
    '--cpus-per-task',
    type=int,
    default=2,
    help='Number of cpus per task (only applicable to launcher == "slurm")')
@click.option(
    '-p',
    '--partition',
    type=str,
    help='The partition to use (only applicable to launcher == "slurm")')
@click.option(
    '-j', '--max-jobs', type=int, help='Max parallel number', default=1)
@click.option(
    '--srun-args', type=str, help='Other srun arguments that might be used')
@click.option('--mj', is_flag=True, help='Multiple Jobs Per Node')
@click.option(
    '-S',
    '--search-args',
    type=str,
    help='Arguments for hyper parameters search')
@click.argument('other_args', nargs=-1, type=click.UNPROCESSED)
def cli(package: str,
        config: str,
        gpus: int,
        gpus_per_node: int,
        partition: str,
        cpus_per_task: int = 2,
        max_jobs: int = 1,
        launcher: str = 'pytorch',
        port: int = None,
        srun_args: Optional[str] = None,
        search_args: str = '',
        mj: bool = True,
        other_args: tuple = ()) -> None:
    """Perform Hyper-parameter search.

    Example:

    \b
    # Parameter search on a single server with CPU by setting `gpus` to 0 and
    # 'launcher' to 'none' (if applicable). The training script of the
    # corresponding codebase will fail if it doesn't support CPU training.
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus \
        0 --search-args '--optimizer.lr 1e-2 1e-3'
    # Parameter search with on a single server with one GPU, search learning
    # rate
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus \
        1 --search-args '--optimizer.lr 1e-2 1e-3'
    # Parameter search with on a single server with one GPU, search
    # weight_decay
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus \
        1 --search-args '--optimizer.weight_decay 1e-3 1e-4'
    # Parameter search with on a single server with one GPU, search learning
    # rate and weight_decay
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus \
        1 --search-args '--optimizer.lr 1e-2 1e-3 --optimizer.weight_decay \
        1e-3 1e-4'
    # Parameter search on a slurm HPC with one 8-GPU node, search learning
    # rate and weight_decay
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus \
        8 --partition partition_name --gpus-per-node 8 --launcher slurm \
        --search-args '--optimizer.lr 1e-2 1e-3 --optimizer.weight_decay 1e-3 \
        1e-4'
    # Parameter search on a slurm HPC with one 8-GPU node, search learning
    # rate and weight_decay, max parallel jobs is 2
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus \
        8 --partition partition_name --gpus-per-node 8 --launcher slurm \
        --max-jobs 2 --search-args '--optimizer.lr 1e-2 1e-3 \
        --optimizer.weight_decay 1e-3 1e-4'
    # Print the help message of sub-command search
    > mim gridsearch -h
    # Print the help message of sub-command search and the help message of the
    # training script of codebase mmcls
    > mim gridsearch mmcls -h
    """

    is_success, msg = gridsearch(
        package=package,
        config=config,
        gpus=gpus,
        gpus_per_node=gpus_per_node,
        cpus_per_task=cpus_per_task,
        max_jobs=max_jobs,
        partition=partition,
        launcher=launcher,
        port=port,
        srun_args=srun_args,
        search_args=search_args,
        mj=mj,
        other_args=other_args)

    if is_success:
        echo_success(msg)  # type: ignore
    else:
        exit_with_error(msg)


def gridsearch(
    package: str,
    config: str,
    gpus: int,
    gpus_per_node: int = None,
    cpus_per_task: int = 2,
    max_jobs: int = 1,
    partition: str = None,
    launcher: str = 'pytorch',
    port: int = None,
    srun_args: Optional[str] = None,
    search_args: str = '',
    mj: bool = True,
    other_args: tuple = ()
) -> Tuple[bool, Union[str, Exception]]:
    """Hyper parameter search with given config.

    Args:
        package (str): The codebase name.
        config (str): The config file path. If not exists, will search in the
            config files of the codebase.
        gpus (int): Number of gpus used for training.
        gpus_per_node (int, optional): Number of gpus per node to use
            (only applicable to launcher == "slurm"). Defaults to None.
        cpus_per_task (int, optional): Number of cpus per task to use
            (only applicable to launcher == "slurm"). Defaults to None.
        partition (str, optional): The partition name
            (only applicable to launcher == "slurm"). Defaults to None.
        max_jobs (int, optional): The max number of workers. Applicable only
            if launcher == 'slurm'. Default to 1.
        launcher (str, optional): The launcher used to launch jobs.
            Defaults to 'pytorch'.
        port (int | None, optional): The port used for inter-process
            communication (only applicable to slurm / pytorch launchers).
            Default to None. If set to None, will randomly choose a port
            between 20000 and 30000.
        srun_args (str, optional): Other srun arguments that might be
            used, all arguments should be in a string. Defaults to None.
        search_args (str, optional): Arguments for hyper parameters search, all
            arguments should be in a string. Defaults to None.
        mj (bool): Multiple jobs per node, only applicable to launcher == 'pytorch'. Default: True.
        other_args (tuple, optional): Other arguments, will be passed to the
            codebase's training script. Defaults to ().
    """
    full_name = module_full_name(package)
    if full_name == '':
        msg = f"Can't determine a unique package given abbreviation {package}"
        raise ValueError(highlighted_error(msg))
    package = full_name

    # If launcher == "slurm", must have following args
    if launcher == 'slurm':
        msg = ('If launcher is slurm, '
               'gpus-per-node and partition should not be None')
        flag = (gpus_per_node is not None) and (partition is not None)
        if not flag:
            raise AssertionError(highlighted_error(msg))

    if not is_installed(package):
        msg = f'You can not train this model without {package} installed.'
        return False, msg

    pkg_root = get_installed_path(package)

    if not osp.exists(config):
        # configs is put in pkg/.mim in PR #68
        config_root = osp.join(pkg_root, '.mim', 'configs')
        if not osp.exists(config_root):
            # If not pkg/.mim/config, try to search the whole pkg root.
            config_root = pkg_root

        # pkg/.mim/configs is a symbolic link to the real config folder,
        # so we need to follow links.
        files = recursively_find(
            pkg_root, osp.basename(config), followlinks=True)

        if len(files) == 0:
            msg = (f"The path {config} doesn't exist and we can not "
                   f'find the config file in codebase {package}.')
            raise ValueError(highlighted_error(msg))
        elif len(files) > 1:
            msg = (
                f"The path {config} doesn't exist and we find multiple "
                f'config files with same name in codebase {package}: {files}.')
            raise ValueError(highlighted_error(msg))

        # Use realpath instead of the symbolic path in pkg/.mim
        config_path = osp.realpath(files[0])
        click.echo(
            f"The path {config} doesn't exist but we find the config file "
            f'in codebase {package}, will use {config_path} instead.')
        config = config_path

    train_script = osp.join(pkg_root, 'tools/train.py')
    if not osp.exists(train_script):
        # A patch only works for pyskl
        train_script = train_script.replace('pyskl/pyskl', 'pyskl')
        assert osp.exists(train_script)

    # parsing search_args
    # the search_args looks like:
    # "--optimizer.lr 0.001 0.01 0.1 --optimizer.weight_decay 1e-4 1e-3 1e-2"
    search_args = search_args.split(';')
    search_args_dicts = [string2args(search_arg) for search_arg in search_args]

    for search_args_dict in search_args_dicts:

        if not len(search_args_dict):
            msg = 'Should specify at least one arg for searching'
            raise ValueError(highlighted_error(msg))

        for k in search_args_dict:
            if search_args_dict[k] is bool:
                msg = f'Should specify at least one value for arg {k}'
                raise ValueError(highlighted_error(msg))

    try:
        from mmengine import Config
    except ImportError:
        try:
            from mmcv import Config
        except ImportError:
            raise ImportError(
                'Please install mmengine to use the gridsearch command: '
                '`mim install mmengine`.')

    cfg = Config.fromfile(config)

    for search_args_dict in search_args_dicts:
        for arg in search_args_dict:
            try:
                arg_value = get_config(cfg, arg)
                if arg_value is not None and not isinstance(arg_value, str):
                    search_args_dict[arg] = [
                        eval(x) for x in search_args_dict[arg]
                    ]
                    for val in search_args_dict[arg]:
                        assert type(val) == type(arg_value)
            except AssertionError:
                msg = f'Arg {arg} not in the config file. '
                raise AssertionError(highlighted_error(msg))

    other_args_dict = string2args(' '.join(other_args))

    # 3 in 1
    if 'all' in other_args_dict:
        other_args_dict.pop('all')
        other_args_dict['validate'] = bool
        other_args_dict['test-last'] = bool
        other_args_dict['test-best'] = bool

    work_dir = other_args_dict.get('work-dir')

    if work_dir:
        work_dir = work_dir[0]
    else:
        work_dir = cfg.get('work_dir')

    if work_dir is None:
        msg = 'work_dir is not specified'
        raise AssertionError(highlighted_error(msg))

    cfg.pop('work_dir', None)

    # remove redundant '/' at the end of work_dir

    assert work_dir  # To pass mypy test
    while work_dir.endswith('/'):
        work_dir = work_dir[:-1]

    config_tmpl, config_suffix = osp.splitext(osp.basename(config))
    work_dir_tmpl = work_dir

    cmds = []
    exp_names = []

    for search_args_dict in search_args_dicts:

        arg_names = [k for k in search_args_dict]
        arg_values = [search_args_dict[k] for k in arg_names]

        for combination in itertools.product(*arg_values):
            cur_cfg = Config(dict(cp.deepcopy(cfg)))
            suffix_list = []

            for k, v in zip(arg_names, combination):
                suffix_list.extend([k.split('.')[-1], str(v)])
                set_config(cur_cfg, k, v)

            name_suffix = '/search_' + '_'.join(suffix_list)
            work_dir = work_dir_tmpl + name_suffix
            os.makedirs(work_dir, exist_ok=True)

            config_name = config_tmpl + config_suffix
            exp_names.append(config_tmpl + name_suffix)
            config_path = osp.join(work_dir, config_name)

            cur_cfg['work_dir'] = work_dir
            if cur_cfg.get('resume_from', None):
                assert 'epoch' in cur_cfg['resume_from']
                cur_cfg['resume_from'] = cur_cfg['resume_from'].replace('/epoch', name_suffix + '/epoch')

            # This exp has been launched before
            if osp.exists(config_path):
                total_epochs = cur_cfg['total_epochs']
                # The experiment has finished
                if osp.exists(osp.join(work_dir, f'epoch_{total_epochs}.pth')):
                    continue

                ckpts = os.listdir(work_dir)
                ckpts = [x for x in ckpts if 'epoch_' in x and '.pth' in x and 'best' not in x]

                if len(ckpts):
                    # resume from the latest ckpt
                    max_epoch = max(
                        [int(x.split('.')[0].split('_')[1]) for x in ckpts])
                    cur_cfg['resume_from'] = osp.join(work_dir, f'epoch_{max_epoch}.pth')

            with open(config_path, 'w') as fout:
                fout.write(cur_cfg.pretty_text)
                fout.flush()
                time.sleep(1)

            other_args_dict_ = cp.deepcopy(other_args_dict)
            # other_args_dict_['work-dir'] = [work_dir]

            other_args_str = args2string(other_args_dict_)

            common_args = ['--launcher', launcher] + other_args_str.split()

            if launcher == 'pytorch':
                cport = rd.randint(20000, 30000) if port is None else port
                time.sleep(0.1)
                cmd = [
                    'python', '-m', 'torch.distributed.launch',
                    f'--nproc_per_node={gpus}', f'--master_port={cport}',
                    train_script, config_path
                ] + common_args
            elif launcher == 'slurm':
                parsed_srun_args = srun_args.split() if srun_args else []
                has_job_name = any([('--job-name' in x) or ('-J' in x)
                                    for x in parsed_srun_args])
                if not has_job_name:
                    job_name = osp.splitext(osp.basename(config_path))[0]
                    parsed_srun_args.append(f'--job-name={job_name}_train')
                cmd = [
                    'srun', '-p', f'{partition}', f'--gres=gpu:{gpus_per_node}',
                    f'--ntasks={gpus}', f'--ntasks-per-node={gpus_per_node}',
                    f'--cpus-per-task={cpus_per_task}', '--kill-on-bad-exit=1'
                ] + parsed_srun_args + ['python', '-u', train_script, config_path
                                        ] + common_args

            cmds.append(cmd)

    time.sleep(1)
    for cmd in cmds:
        click.echo(' '.join(cmd))

    tstr = datetime.now().strftime('%y%m%d_%H%M%S_%f')
    succeed_list, fail_list = [], []
    if launcher == 'pytorch':
        num_bash = 1
        if mj:
            num_bash = 8 // gpus

        for i in range(num_bash):
            lines = ['#!/usr/bin/env bash']
            for cmd, exp_name in zip(cmds[i::num_bash], exp_names[i::num_bash]):
                my_gpus = list(range(i * gpus, i * gpus + gpus))
                prefix = f'CUDA_VISIBLE_DEVICES={",".join([str(x) for x in my_gpus])}'
                
                if not mj and os.environ.get('CUDA_VISIBLE_DEVICES', None) is not None:
                    prefix = f'CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}'

                cmd_text = ' '.join([prefix] + cmd)
                click.echo(f'Training command for exp {exp_name} is {cmd_text}. ')
                lines.append(cmd_text)

            with open(f'tmp_{i}_{tstr}.sh', 'w') as fout:
                fout.write('\n'.join(lines))

            if not mj:
                os.system(f'bash tmp_{i}_{tstr}.sh')

        if mj: 
            lines = ['#!/usr/bin/env bash']
            lines.extend([f'bash tmp_{i}_{tstr}.sh &' for i in range(num_bash)])
            with open(f'tmp_{tstr}.sh', 'w') as fout:
                fout.write('\n'.join(lines))
            os.system(f'bash tmp_{tstr}.sh')

    elif launcher == 'slurm':
        if port is not None:
            with Executor(max_workers=max_jobs) as executor:
                for exp, ret in zip(exp_names,
                                    executor.map(subprocess.check_call, cmds)):
                    if ret == 0:
                        click.echo(f'Exp {exp} finished successfully.')
                        succeed_list.append(exp)
                    else:
                        echo_error(f'Exp {exp} not finished successfully.')
                        fail_list.append(exp)
        else:
            for cmd in cmds:
                cmd_str = ' '.join(cmd)
                port = rd.randint(20000, 30000)
                cmd_str = f'MASTER_PORT={port} ' + cmd_str + ' &'

                # usage = get_usage()
                # while usage[0] >= 8 or usage[1] >= 8:
                #     time.sleep(30)
                #     usage = get_usage()

                os.system(cmd_str)
                time.sleep(5)

            return True, 'MIM Random Port Slurm GSearch Finished. '

    if len(fail_list):
        msg = ('The following experiments in hyper parameter search '
               f'failed: \n {fail_list}')
        return False, msg
    else:
        msg = ('The hyper parameter search finished successfully.'
               f'Experiment list: \n {succeed_list}')
        return True, msg
