from inlet.inlet import carToPolar
from math import pi
import numpy as np

import pytest

def test_carToPolar():
    assert carToPolar(2**.5,2**.5) == [2,pi/4]

def test_carToPolar2():
    arg1 = np.array([2**.5])
    arg2 = np.array([2**.5])
    assert carToPolar(arg1,arg2) == [2,pi/4]


