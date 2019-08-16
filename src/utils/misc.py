import os
from dotenv import find_dotenv, load_dotenv
import logging

__all__ = [
    'dot_env_stuff'
]


def dot_env_stuff(func):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    load_dotenv(find_dotenv())
    
    func()
