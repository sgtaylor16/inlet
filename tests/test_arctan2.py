from inlet.inlet import arctan2
import pytest
from math import pi

def test_arctan21():
    assert arctan2([0,1]) == pi/2

def test_arctan22():
    assert arctan2([0,-1]) == 3*pi/2

def test_arctan23():
    assert arctan2([1,0]) == 0.0

def test_arctan24():
    assert arctan2([-1,0]) == pi

def test_arctan25():
    assert arctan2([1,1]) == pi/4

def test_arctan26():
    assert arctan2([-1,1]) == 3*pi/4

def test_arctan27():
    assert arctan2([-1,-1]) == 5*pi/4

def test_arctan28():
    assert arctan2([1,-1]) == 7*pi/4

def test_arctan29():
    assert arctan2([0,0]) == 0.0