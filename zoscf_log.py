"""
Example zero offset shift compensation function file
"""
import math


def calc_zos(ch: int, vscale: float, dtime: float)->float:
    assert vscale == 1.0

    def exp(x, a, b, c):
        return a * math.e**(b * x) + c

    return 0
