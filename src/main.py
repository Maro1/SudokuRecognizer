import numpy as np
import cv2
import matplotlib.pyplot as plt

import math
import os

from digit_model import DigitModel
import sudoku_solver

GRID_SIZE = 900


def plot_image(image: np.ndarray):
    """
    Plots a single image
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.show()


def transform_grid(image: np.ndarray, corners: np.ndarray,
                   grid_size: int = GRID_SIZE) -> tuple:
    """
    Transforms image grid points to standardized grid
    using inverse perspective projection
    """

    # Destination points in clockwise order from top-left corner
    dst_points = np.array(
        [[0, 0],   [grid_size, 0],  [grid_size, grid_size], [0, grid_size]],
        dtype=np.float32)

    perspective_mat = cv2.getPerspectiveTransform(
        np.array(corners, dtype=np.float32), dst_points)

    # For use when projecting numbers back onto image
    try:
        inv_perspective = np.linalg.inv(perspective_mat)
    except Exception:
        return None, None

    warped = cv2.warpPerspective(
        image, perspective_mat, (grid_size, grid_size))

    return warped, inv_perspective


def sort_contour(contour: np.ndarray) -> np.ndarray:
    """
    Sorts 4 corner contour in clockwise order from top-left corner
    """

    # OpenCV contours have one redundant dimension, using only 2 for
    # easier sorting
    new_contour = []
    for c in contour:
        new_contour.append(c[0])
    new_contour = np.array(new_contour)

    # Find the center of the contour
    center_x = sum([x[0] for x in new_contour]) / len(new_contour)
    center_y = sum([x[1] for x in new_contour]) / len(new_contour)

    # Sort contour points in clockwise order starting with top-left corner
    for c in contour:
        point = c[0]
        if point[0] < center_x and point[1] < center_y:
            new_contour[0] = point
        elif point[0] > center_x and point[1] < center_y:
            new_contour[1] = point
        elif point[0] > center_x and point[1] > center_y:
            new_contour[2] = point
        elif point[0] < center_x and point[1] > center_y:
            new_contour[3] = point

    # Convert back to OpenCV contour format
    contour = np.array([[x] for x in new_contour])
    return contour


def find_sudoku_corners(image: np.ndarray) -> np.ndarray:
    """
    Finds the four corners of sudoku grid using contours
    """

    # Finds contours in binary image using Suzuki85
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If less than the grid size of sudoku contours are found,
    # there is no sudoku board
    if len(contours) < 9 * 9:
        return None

    # Find second largest contour which should be the outermost
    # contour of sudoku board
    contour = sorted(contours, key=lambda c: -cv2.contourArea(c))[1]

    # The contour is approximated as a rectangle
    contour = cv2.approxPolyDP(
        contour, 0.01*cv2.arcLength(contour, True), True)

    if len(contour) != 4:
        return None

    contour = sort_contour(contour)
    return contour


def find_grid(sudoku_img: np.ndarray) -> tuple:
    """
    Finds standardized sudoku grid from input image
    """

    # Apply gaussian filter to de-noise
    blurred_img = cv2.GaussianBlur(sudoku_img, (7, 7), 3)

    # Use adaptive threshholding to better handle varying lighting in image
    thresh = cv2.adaptiveThreshold(
        blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 7, 2)

    contour_corners = find_sudoku_corners(thresh)

    if contour_corners is None:
        return None, None

    transformed, inv_perspective = transform_grid(sudoku_img, contour_corners)
    if transformed is None:
        return None, None

    return transformed, inv_perspective


def find_cells(grid: np.ndarray) -> list:
    """
    Finds the individual cells of the transformed sudoku images
    and returns them as a list
    """
    cells = []

    # Split the image into 9 rows
    row = np.vsplit(grid, 9)

    for col in row:

        # Split the row into 9 columns
        col_cells = np.hsplit(col, 9)

        for cell in col_cells:

            # Apply a binary threshold using Otsu's to the cell
            _, cell = cv2.threshold(
                cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Invert cell and take only central part to exclude edges
            cell = ~cell[15:85, 15:85]

            # If too many or too few pixels are non-zero, add empty cell
            if np.count_nonzero(cell) > cell.size / 2:
                return None
            if np.count_nonzero(cell) < 5:
                cells.append(None)
            else:
                cells.append(cell)

    return cells


def get_model() -> DigitModel:
    """
    Loads model from weights if they exists,
    otherwise train model
    """
    if os.path.exists(os.path.join(os.getcwd(), 'model/checkpoint')):
        model = DigitModel('model/checkpoint')
    else:
        model = DigitModel()
        model.train(epochs=100)

    return model


def classify_grid(model: DigitModel, cells: list) -> np.ndarray:
    """
    Iterates over each cell in grid and classifies digit if non empty
    """

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
        # De-noise with Gaussian blur
        cell = cv2.GaussianBlur(cell, (5, 5), 13)
        # Resize image to same as dataset used to train model
        cell = cv2.resize(cell, (28, 28))
        cell = cell.astype(np.float32)  # Use float type
        cell /= 255.  # Divide by 255 to get float between 0 and 1

        to_classifiy.append(cell)
        idx_dict[idx] = math.floor(i / 9), i % 9
        idx += 1

    if not to_classifiy:
        return None

    numbers = model.classify_number(np.asarray(to_classifiy))

    for i in range(len(numbers)):
        # Update sudoku grid with classified number
        sudoku_grid[idx_dict[i]] = numbers[i]

    return sudoku_grid


def project_solution(img: np.ndarray, sudoku_grid: np.ndarray,
                     solved_grid: np.ndarray,
                     inv_perspective: np.ndarray) -> np.ndarray:
    """
    Projects solved sudoku digits onto image using 
    inverse perspective projection
    """

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
                    int(solved_grid[i, j])), (x, y), cv2.FONT_HERSHEY_COMPLEX,
                    2, (255, 255, 100))

    # Apply inverse perspective homography to get digits in
    # original image space
    solution = cv2.warpPerspective(
        solution, inv_perspective, (img.shape[1], img.shape[0]))

    # Since all non-digit pixels of solution are 0, subtract
    # can be used to overlay
    return cv2.subtract(img, solution)


def run(path: str):
    # Digit classifier model
    model = get_model()

    # Get image both in RGB and grayscale
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get sudoku grid image along with inverse
    # perspective transformation
    sudoku_img, inv_perspective = find_grid(gray_img)

    # If not found, plot original image
    if sudoku_img is None:
        plot_image(img)
        return

    cells = find_cells(sudoku_img)
    if cells is None:
        plot_image(img)
        return

    sudoku_grid = classify_grid(model, cells)
    if sudoku_grid is None:
        plot_image(img)
        return

    # Copy grid to only overlay solved digits later
    solved_grid = sudoku_grid.copy()
    if sudoku_solver.valid(solved_grid):
        sudoku_solver.solve(solved_grid)

        to_present = project_solution(
            img, sudoku_grid, solved_grid, inv_perspective)
        plot_image(to_present)
    else:
        plot_image(img)


def run_video(path: str, save=False):
    model = get_model()

    cap = cv2.VideoCapture(path)

    # Keep track of solved grid to not have to
    # solve each frame
    solved_grid = None

    # If saving video, create image list
    if save:
        img_list = []

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break

        if cv2.waitKey(1) == ord('q'):
            break

        if save:
            img_list.append(img)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sudoku_img, inv_perspective = find_grid(gray_img)
        if sudoku_img is None:
            solved_grid = None
            cv2.imshow('frame', cv2.resize(img, (500, 900)))
            continue

        cells = find_cells(sudoku_img)
        if cells is None:
            solved_grid = None
            cv2.imshow('frame', cv2.resize(img, (500, 900)))
            continue

        sudoku_grid = classify_grid(model, cells)
        if sudoku_grid is None:
            solved_grid = None
            cv2.imshow('frame', cv2.resize(img, (500, 900)))
            continue

        if sudoku_solver.valid(sudoku_grid):
            if solved_grid is None:
                # Copy grid to only overlay solved digits later
                solved_grid = sudoku_grid.copy()
                sudoku_solver.solve(solved_grid)

            if solved_grid.all():
                to_present = project_solution(
                    img, sudoku_grid, solved_grid, inv_perspective)
                cv2.imshow('frame', cv2.resize(to_present, (500, 900)))

                if save:
                    img_list[-1] = to_present
                continue

        cv2.imshow('frame', cv2.resize(img, (500, 900)))

    cap.release()
    cv2.destroyAllWindows()

    # Save video
    if save:
        out = cv2.VideoWriter(
            'produced_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
            30, (1080, 1920))

        for i in range(len(img_list)):
            out.write(img_list[i])

        out.release()


if __name__ == '__main__':
    print('-------- SudokuRecognizer --------')
    print('1. Run on image')
    print('2. Run on video')
    choice = input('Please choose an option: ')
    try:
        choice = int(choice)
        if choice < 1 or choice > 2:
            raise Exception()
    except Exception:
        print('Unknown option, exiting...')
        exit(-1)

    if choice == 1:
        path = input('Please enter image path: ')
        if not os.path.exists(path):
            print('Unknown path, exiting...')
            exit(-1)
        run(path)

    elif choice == 2:
        path = input('Please enter video path: ')
        if not os.path.exists(path):
            print('Unknown path, exiting...')
            exit(-1)
        run_video(path, False)
