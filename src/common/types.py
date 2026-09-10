from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Observation:
    depth_fan: np.ndarray #left to right across FOV (meters)
    goal_vector: np.ndarray #goal position relative to the agent (meters)
    heading: float #radians, agent frames
    
    def to_array(self) -> np.ndarray:
        return np.concatenate([self.depth_fan, self.goal_vector, [self.heading]])
    
@dataclass(frozen=True)
class GuidanceAction:
    turn: float #radians, positive = right
    speak: bool #output or not