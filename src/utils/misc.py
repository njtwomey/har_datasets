import os
from dotenv import find_dotenv, load_dotenv
import logging
import zipfile

__all__ = [
    'dot_env_stuff', 'unzip_data',
]


def dot_env_stuff(func):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    load_dotenv(find_dotenv())
    
    func()


def unzip_data(zip_path, in_name, out_name):
    if os.path.exists(os.path.join(zip_path, out_name)):
        return
    with zipfile.ZipFile(os.path.join(zip_path, in_name), 'r') as fil:
        fil.extractall(zip_path)
