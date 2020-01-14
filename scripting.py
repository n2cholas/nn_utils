# Utilities for creating scripts.

import argparse
import os
from typing import Dict, Any, Optional
import warnings

_ARGDICT = {
  'lr':      {'help': 'Learning Rate', 'type': float},
  'wd':      {'help': 'Weight Decay', 'type': float},
  'bs':      {'help': 'Batch Size', 'type': float},
  'steps':   {'help': 'Num Steps', 'type': float},
  'mom':     {'help': 'Momentum', 'type': float},
  'maxlr':   {'help': 'Maximum Learning Rate', 'type': float},
  'maxmom':  {'help': 'Maximum Momentum', 'type': float},
  'basemom': {'help': 'Base Momentum', 'type': float},
  'seqlen':  {'help': 'Sequence Length', 'type': int},
  'device':  {'help': 'Device Name', 'type': str},
}

def get_argparser(**kwargs) -> argparse.ArgumentParser:
  """Pass in the required flags with their defaults."""
  parser = argparse.ArgumentParser()
  for arg, default in kwargs.items():
    parser.add_argument(f'-{arg}', default=default, **_ARGDICT.get(arg, {}))
    if arg not in _ARGDICT:
      warnings.warn(f'nn_utils.scripting.py does not recongize '
                    '{arg} as a default command line argument.')
  return parser

def get_num_runs(log_dir: str='logs') -> int:
  # Assumes runs were produced by training.py (each run has a one folder).
  try:
    return len(list(os.listdir(log_dir)))
  except FileNotFoundError:
    return 0

def get_log_path(log_dir: str='logs', 
                 run_info: Optional[Dict[str, Any]]=None,
                 other_info: Optional[str]=None) -> str:
  run_info = run_info or {}
  other_info = '' if not other_info else f'_{other_info}'
  log_name = f'run{get_num_runs(log_dir):02d}{other_info}'
  log_name = log_name + ''.join([f'_{k}={v}' for k, v in run_info.items()])
  return os.path.join(log_dir, log_name)