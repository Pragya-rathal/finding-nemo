import time
from contextlib import contextmanager

@contextmanager
def timer(name: str='block'):
    st=time.time(); yield; et=time.time()
    print(f'{name}: {et-st:.4f}s')
