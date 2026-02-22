import shutil
from local import Config

config = Config()
path_save_codes = f'{config.path_model}/model_codes'

shutil.copy(f'{path_save_codes}/train.py', './train.py')
shutil.copytree(f'{path_save_codes}/data', './data', dirs_exist_ok=True)
shutil.copytree(f'{path_save_codes}/model', './model',ignore=shutil.ignore_patterns('DCNv2*'), dirs_exist_ok=True)
shutil.copy(f'{path_save_codes}/local.py', './local.py')
