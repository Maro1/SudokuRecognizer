import numpy as np
import cv2
import matplotlib.pyplot as plt

import math
import os

from digit_model import DigitModel
import sudoku_solver


DEBUG = False
GRID_SIZE = 900

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

    # For use when projecting numbers back onto image
    inv_perspective = np.linalg.inv(perspective_mat)

    warped = cv2.warpPerspective(
        image, perspective_mat, (grid_size, grid_size))

    return warped, inv_perspective


def find_sudoku_corners(image):
    """
    Finds the four corners of sudoku grid using contours
    """

    # Finds contours in binary image using Suzuki85
    contours, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find second largest contour and approximate it as rectangle
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

    transformed, inv_perspective = transform_grid(sudoku_img, contour_corners)

    return transformed, inv_perspective


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

def get_model() -> DigitModel:
    """
    Loads model from weights if they exists,
    otherwise trains model
    """
    if os.path.exists(os.path.join(os.getcwd(), 'model/checkpoint')):
        model = DigitModel('model/weights.hdf')
    else:
        model = DigitModel()
        model.train(epochs=100)

    return model

def classify_grid(model: DigitModel, cells: list) -> np.ndarray:
    """
    Iterates over each cell in grid and classifies digit if non empty
    """

    # TODO: Classify all numbers at once to improve performance
    sudoku_grid = np.zeros((9, 9))

    to_classifiy = []
    idx_dict = {}
    idx = 0

    for i in range(len(cells)):
        cell = cells[i]

        # No need to classify empty cells
        if cell is None:
            continue

        # Prepare data for trained model
        cell = cv2.GaussianBlur(cell, (5, 5), 13) # De-noise with Gaussian blur
        cell = cv2.resize(cell, (28, 28)) # Resize image to same as dataset used to train model
        cell = cell.astype(np.float32) # Use float type
        cell /= 255. # Divide by 255 to get float between 0 and 1

        to_classifiy.append(cell)
        idx_dict[idx] = math.floor(i / 9), i % 9
        idx += 1

    numbers = model.classify_number(np.asarray(to_classifiy))

    for i in range(len(numbers)):
        sudoku_grid[idx_dict[i]] = numbers[i] # Update sudoku grid with classified number
    
    return sudoku_grid

def project_solution(img: np.ndarray, sudoku_grid: np.ndarray, solved_grid: np.ndarray, inv_perspective: np.ndarray) -> np.ndarray:
    """
    Projects solved sudoku digits onto image using inverse perspective projection
    """

    # Convert to BGR to project digits in color
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
    
    # Init solution image and cell size
    cell_size = int(GRID_SIZE / 9)
    solution = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)

    for i in range(9):
        for j in range(9):
            if sudoku_grid[i, j] == 0:
                # Find x and y for text based on cell location
                x = int(j * cell_size + 0.3 * cell_size)
                y = int(i * cell_size + 0.8 * cell_size)

                cv2.putText(solution, str(
                    int(solved_grid[i, j])), (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 100))

    # Apply inverse perspective homography to get digits in original image space
    solution = cv2.warpPerspective(solution, inv_perspective, (img.shape[1], img.shape[0]))

    # Since all non-digit pixels of solution are 0, subtract can be used to overlay
    return cv2.subtract(img, solution)

def run():
    model = get_model()

    img = cv2.imread('images/sudoku_5.jpg', cv2.IMREAD_GRAYSCALE)
    sudoku_img, inv_perspective = find_grid(img)

    cells = find_cells(sudoku_img)

    sudoku_grid = classify_grid(model, cells)

    # Copy grid to only overlay solved digits later
    solved_grid = sudoku_grid.copy()
    sudoku_solver.solve(solved_grid)

    to_present = project_solution(img, sudoku_grid, solved_grid, inv_perspective)
    plot_image(to_present)

def run_video():
    model = get_model()

    cap = cv2.VideoCapture('images/Sudoku2.mp4')

    first_sudoku = 0
    prev_solved_grid = None
    prev_sudoku_grid = None
    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            sudoku_img, inv_perspective = find_grid(img)
            cells = find_cells(sudoku_img)
        except Exception as e:
            first_sudoku = False
            cv2.imshow('frame', cv2.resize(img, (500, 900)))
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        if not first_sudoku:
            first_sudoku = True
            sudoku_grid = classify_grid(model, cells)

            if sudoku_solver.valid(sudoku_grid):
                # Copy grid to only overlay solved digits later
                solved_grid = sudoku_grid.copy()
                sudoku_solver.solve(solved_grid)

                if solved_grid.all():
                    prev_sudoku_grid = sudoku_grid
                    prev_solved_grid = solved_grid

        if prev_solved_grid is not None:
            to_present = project_solution(img, prev_sudoku_grid, prev_solved_grid, inv_perspective)
            cv2.imshow('frame', cv2.resize(to_present, (500, 900)))

        else:
            cv2.imshow('frame', cv2.resize(img, (500, 900)))

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #import cProfile
    #cProfile.run('run_video()', 'restats')
    run_video()
