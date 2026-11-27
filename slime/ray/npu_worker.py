"""Signal setup for NPU workers started with the memory-saver preload library."""

import signal
import sys
from multiprocessing import forkserver


def _restore_python_sigint():
    # A native handler installed before Python starts is reported as None.
    # Ray and multiprocessing both save and later reinstall the Python value.
    if signal.getsignal(signal.SIGINT) is None:
        signal.signal(signal.SIGINT, signal.default_int_handler)


def setup():
    _restore_python_sigint()
    # Each forkserver starts a fresh interpreter with the same LD_PRELOAD.
    modules = forkserver._forkserver._preload_modules
    if __name__ not in modules:
        forkserver.set_forkserver_preload([*modules, __name__])


if any(arg.startswith("from multiprocessing.forkserver import main;") for arg in sys.orig_argv):
    _restore_python_sigint()
