import numpy as np
import cv2
import matplotlib.pyplot as plt

import math

from mnist_model import MNISTModel

DEBUG = False
GRID_SIZE = 900
USE_WEIGHTS = True


def plot_image(image):
    """
    Plots a single image
    """
    plt.imshow(image, cmap='gray')
    plt.show()


def transform_grid(image, corners, grid_size=GRID_SIZE):
    """
    Transforms image grid points to standardized grid
    using inverse perspective projection
    """

    dst_points = np.array(
        [[grid_size, 0],   [grid_size, grid_size],  [0, grid_size], [0, 0]], dtype=np.float32)

    perspective_mat = cv2.getPerspectiveTransform(
        np.array(corners, dtype=np.float32), dst_points)

    warped = cv2.warpPerspective(
        image, perspective_mat, (grid_size, grid_size))

    return warped


def find_sudoku_corners(image):
    """
    Finds the four corners of sudoku grid using contours
    """

    # Finds contours in binary image using Suzuki85
    contours, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find second largest contour and find approximate it as rectangle
    contour = sorted(contours, key=lambda c: -cv2.contourArea(c))[1]
    contour = cv2.approxPolyDP(
        contour, 0.01*cv2.arcLength(contour, True), True)

    return contour


def find_grid(sudoku_img):
    """
    Finds standardized sudoku grid from input image
    """

    # Convert to grayscale and de-noise
    grayscale_img = cv2.GaussianBlur(sudoku_img, (7, 7), 3)

    # Use adaptive threshholding to better handle varying lighting in image
    #! NOTE: Instead of 11, 2, make it depend on image resolution?
    thresh = cv2.adaptiveThreshold(
        grayscale_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contour_corners = find_sudoku_corners(thresh)

    if DEBUG:
        cv2.drawContours(
            sudoku_img, [contour_corners], -1, (255, 0, 0), thickness=5)

    transformed = transform_grid(sudoku_img, contour_corners)

    return transformed


def find_cells(grid):
    cell_size = int(GRID_SIZE / 9)
    offset = 12
    cells = []

    row = np.vsplit(grid, 9)

    for col in row:
        col_cells = np.hsplit(col, 9)

        for cell in col_cells:
            _, cell = cv2.threshold(
                cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cell = ~cell[15:85, 15:85]

            if np.count_nonzero(cell) < 5:
                cells.append(None)
            else:
                contours, _ = cv2.findContours(
                    cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour = max(contours, key=lambda c: cv2.contourArea(c))

                min_x = cell_size
                min_y = cell_size
                max_x = 0
                max_y = 0
                for p in contour:
                    if p[0][0] > max_x:
                        max_x = p[0][0]
                    if p[0][0] < min_x:
                        min_x = p[0][0]
                    if p[0][1] > max_y:
                        max_y = p[0][1]
                    if p[0][1] < min_y:
                        min_y = p[0][1]

                cell = cell[min_y:max_y, min_x:max_x]

                y = max_y - min_y
                x = max_x - min_x
                size = y + int(y * 0.3) + (y + int(y * 0.3)) % 2
                padding_y = int((size - y) / 2)
                padding_x = int((size - x) / 2)

                cell = cv2.copyMakeBorder(
                    cell, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT, value=0)

                cells.append(cell)

    return cells


img = cv2.imread('images/sudoku_4.jpg', cv2.IMREAD_GRAYSCALE)
sudoku_img = find_grid(img)

sudoku_grid = np.zeros((9, 9))
cells = find_cells(sudoku_img)

if USE_WEIGHTS:
    model = MNISTModel('model/weights.hdf')
else:
    model = MNISTModel()
    model.train()

for i in range(len(cells)):
    cell = cells[i]
    if cell is None:
        continue

    # Prepare data for trained MNIST model
    cell = cv2.resize(cell, (28, 28))
    cell = cell.astype(np.float32)
    cell /= 255.

    number = model.classify_number(cell)
    sudoku_grid[math.floor(i / 9), i % 9] = number

print(sudoku_grid)
plot_image(img)
