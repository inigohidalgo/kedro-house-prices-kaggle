from contextlib import contextmanager
import os


@contextmanager
def set_cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)
