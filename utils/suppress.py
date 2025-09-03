import builtins
import sys
import os
import logging
import wandb


def suppress_print():
    """Suppresses printing from the current process."""
    def ignore(*_objects, _sep=" ", _end="\n", _file=sys.stdout, _flush=False):
        pass
    builtins.print = ignore


def suppress_wandb():
    """Suppresses wandb logging from the current_process."""
    # Store original functions
    original_functions = {}
    for attr_name in dir(wandb):
        attr = getattr(wandb, attr_name)
        if callable(attr) and not attr_name.startswith('__'):
            original_functions[attr_name] = attr

            # Replace with no-op function
            def make_noop(name):
                def noop(*args, **kwargs):
                    pass
                return noop

            setattr(wandb, attr_name, make_noop(attr_name))


def suppress_logging():
    """Suppresses logging from the current process."""
    logging.getLogger().setLevel(logging.CRITICAL + 1)  # Above CRITICAL level