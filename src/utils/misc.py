import os
import tempfile
from dotenv import find_dotenv, load_dotenv
import logging
import zipfile

__all__ = [
    'dot_env_stuff', 'unzip_data', 'symlink_allowed', 'make_symlink'
]


def dot_env_stuff(func):
    """
    
    :param func:
    :return:
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    load_dotenv(find_dotenv())
    
    func()


def unzip_data(zip_path, in_name, out_name):
    """
    
    :param zip_path:
    :param in_name:
    :param out_name:
    :return:
    """
    if os.path.exists(os.path.join(zip_path, out_name)):
        return
    with zipfile.ZipFile(os.path.join(zip_path, in_name), 'r') as fil:
        fil.extractall(zip_path)


def symlink_allowed():
    """
    Symbolic link permissions are not allowed by default with some
    operating systems (eg Windows 10).
    :return:
    """
    try:
        with tempfile.NamedTemporaryFile() as fil:
            os.symlink(fil.name, fil.name + '-symlink')
    except OSError:
        return False
    return True


def make_symlink(in_path, out_path, ext):
    """
    
    :param in_path:
    :param out_path:
    :param ext:
    :return:
    """
    in_path = f'{in_path}.{ext}'
    out_path = f'{out_path}.{ext}'
    assert os.path.exists(in_path)
    if not os.path.exists(out_path):
        os.symlink(in_path, out_path)
