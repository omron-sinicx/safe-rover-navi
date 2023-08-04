"""
description: define motion model for the planning algorithm
author: Masafumi Endo
"""

def motion_model():
    """
    motion_model: define motion model as go forward eight directions

    """
    motion = [[1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1]]
    return motion