"""
Context-based regularizers, thread-safe

Usage:
>>> with collect_regularizers() as regularizers:
>>>   y1 = model1(x)
>>>   y2 = model2(x)
>>>
>>> do_something([regularizers[module]['my_activation'] for module in regularizers])

Inside model1 and model2 code:
>>> <...>
>>> activation = self.foo(x)
>>> if is_regularized('my_activation')
>>>   add_regularizer(self, 'my_activation', activation)
>>> output = self.bar(activation)
>>> return output

"""
from contextlib import contextmanager
from collections import defaultdict
import threading
from warnings import warn


REGULARIZERS_KEYS = None
REGULARIZERS = None
tls = threading.local()


@contextmanager
def collect_regularizers(collection=None, keys=None, within_thread=None):
    if within_thread is None:
        if threading.current_thread() is not threading.main_thread():
            warn("Calling collect_regularizers while not in main thread, please set within_thread explicitly")
        within_thread = threading.current_thread() == threading.main_thread()

    if collection is None:
        collection = defaultdict(lambda: defaultdict(list))

    global REGULARIZERS, REGULARIZERS_KEYS
    setattr(tls, 'REGULARIZERS', getattr(tls, 'REGULARIZERS', None))
    setattr(tls, 'REGULARIZERS_KEYS', getattr(tls, 'REGULARIZERS_KEYS', None))

    _old_regs, _old_keys = REGULARIZERS, REGULARIZERS_KEYS
    _old_local_regs, _old_local_keys = tls.REGULARIZERS, tls.REGULARIZERS_KEYS
    try:
        if within_thread:
            tls.REGULARIZERS, tls.REGULARIZERS_KEYS = collection, keys
            REGULARIZERS, REGULARIZERS_KEYS = None, None
        else:
            REGULARIZERS, REGULARIZERS_KEYS = collection, keys
            tls.REGULARIZERS, tls.REGULARIZERS_KEYS = None, None

        yield collection
    finally:
        REGULARIZERS = _old_regs
        REGULARIZERS_KEYS = _old_keys
        tls.REGULARIZERS = _old_local_regs
        tls.REGULARIZERS_KEYS = _old_local_keys


def get_regularized_keys():
    is_local = hasattr(tls, 'REGULARIZERS')
    return getattr(tls, 'REGULARIZERS_KEYS', REGULARIZERS_KEYS) if is_local else REGULARIZERS_KEYS


def get_regularizer_collection():
    is_local = hasattr(tls, 'REGULARIZERS')
    return getattr(tls, 'REGULARIZERS', None) if is_local else REGULARIZERS


def is_regularized(key):
    if get_regularizer_collection() is None:
        return False
    keys = get_regularized_keys()
    return keys is None or key in keys


def add_regularizer(module, key, value):
    assert is_regularized(key)
    get_regularizer_collection()[module][key].append(value)
