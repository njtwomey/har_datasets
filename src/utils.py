import os
from dotenv import find_dotenv, load_dotenv
import logging

from numpy import arange
import pandas as pd

import zipfile
import yaml


def load_data(fname, astype='list'):
    df = pd.read_csv(
        fname,
        delim_whitespace=True,
        header=None
    )
    if astype in {'dataframe', 'pandas', 'pd'}:
        return df
    if astype in {'values', 'np', 'numpy'}:
        return df.values
    if astype == 'list':
        return df.values.tolist()
    
    raise ValueError


def unzip_data(zip_path, in_name, out_name):
    if os.path.exists(os.path.join(zip_path, out_name)):
        return
    with zipfile.ZipFile(os.path.join(zip_path, in_name), 'r') as fil:
        fil.extractall(zip_path)


def build_time(subs, win_len, fs):
    win = arange(win_len, dtype=float) / fs
    inc = win_len / fs
    t = []
    prev_sub = subs[0]
    for curr_sub in subs:
        if curr_sub != prev_sub:
            win = arange(win_len, dtype=float) / fs
        t.extend(win)
        win += inc
        prev_sub = curr_sub
    return t


def build_seq_list(subs, win_len):
    seq = []
    si = 0
    last_sub = subs[0]
    for prev_sub in subs:
        if prev_sub != last_sub:
            si += 1
        seq.extend([si] * win_len)
        last_sub = prev_sub
    return seq


def dot_env_stuff(func):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    load_dotenv(find_dotenv())
    
    func()


def load_datasets():
    return yaml.load(open(os.path.join(
        os.environ['PROJECT_ROOT'], 'datasets.yaml'
    ), 'r'))


def load_activities():
    return yaml.load(open(os.path.join(
        os.environ['PROJECT_ROOT'], 'activities.yaml'
    ), 'r'))


def check_activities(acts):
    all_activities = load_activities()
    for act in acts:
        if act not in all_activities:
            raise ValueError(f'{act} is not yet in `activities.yaml`')
    return {vv: kk for kk, vv in acts.items()}


def standardise_activities(lookup, activities):
    return [lookup[act] for act in activities]


def module_importer(class_name, *args, **kwargs):
    m = __import__('src.processors', fromlist=[class_name])
    c = getattr(m, class_name)
    return c(*args, **kwargs)
