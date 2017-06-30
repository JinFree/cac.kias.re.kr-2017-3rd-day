from numba import jit

@jit
def test(msg):
    return msg + '?'

print test('hello')
