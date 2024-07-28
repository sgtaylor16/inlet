from pydantic import BaseModel
from typing import Literal, List

class OneProbe(BaseModel):
    name:str
    r: float
    theta: float

class RakePositions(BaseModel):
    rakepositions: List[OneProbe]
    OD: float

class DataPoint(BaseModel):
    name:str
    yaw: float
    pitch: float
    pt: float