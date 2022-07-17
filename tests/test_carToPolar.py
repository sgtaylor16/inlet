from inlet.inlet import carToPolar
from math import pi
import pytest

def test_carToPolar1():
    assert carToPolar([1,0]) == [1,0]

def test_carToPolar2():
    assert carToPolar([0,1]) == [1,pi/2]

def test_carToPolar3():
    assert carToPolar([-1,0]) == [1,pi]

def test_carToPolar4():
    assert carToPolar([0,-1]) == [1,1.5*pi]

def test_carToPolar5():
    assert carToPolar([1,1]) == [2**.5,pi/4]

def test_carToPolar6():
    assert carToPolar([-1,1]) == [2**.5,3*pi/4]