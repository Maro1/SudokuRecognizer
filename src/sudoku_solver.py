import numpy as np
import math

from numpy.core.fromnumeric import nonzero


def solve(grid: np.ndarray, pos: tuple = (0, 0)) -> bool:
    while grid[pos[0], pos[1]] != 0:
        pos = next_pos(pos)
        if pos == None:
            return True

    found_valid = False
    for num in range(1, 10):
        if valid_num(grid, pos, num):
            grid[pos[0], pos[1]] = num

            if next_pos(pos) == None:
                return True

            if solve(grid, next_pos(pos)):
                return True
            else:
                grid[pos[0], pos[1]] = 0

    return False


def valid_num(grid: np.ndarray, pos: tuple, num: int) -> bool:
    # Vertical and horizontal
    if num in grid[pos[0], :] or num in grid[:, pos[1]]:
        return False

    # 3x3 box
    box_x = 3 * math.floor(pos[1] / 3)
    box_y = 3 * math.floor(pos[0] / 3)
    if num in grid[box_y:box_y + 3, box_x:box_x + 3]:
        return False

    return True


def next_pos(pos: tuple):
    if pos[1] < 8:
        return (pos[0], pos[1] + 1)
    elif pos[0] < 8:
        return (pos[0] + 1, 0)
    else:
        return None

def _nonzero(x):
    return [n for n in x if n != 0]

def valid(grid: np.ndarray):
    for i in range(9):
        for j in range(9):
            if grid[i, j] == 0:
                continue

            if len(set(_nonzero(grid[i, :]))) != len(_nonzero(grid[i, :])):
                return False
            if len(set(_nonzero(grid[:, j]))) != len(_nonzero(grid[:, j])):
                return False

            box_x = 3 * math.floor(j / 3)
            box_y = 3 * math.floor(i / 3)
            if len(set(_nonzero(grid[box_y:box_y + 3, box_x:box_x + 3].ravel()))) != len(_nonzero(grid[box_y:box_y + 3, box_x:box_x + 3].ravel())):
                return False

    return True

