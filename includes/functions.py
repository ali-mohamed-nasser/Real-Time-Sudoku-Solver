# Importing needed libraries and files
import cv2 as cv
import numpy as np
from scipy import ndimage
import math
from includes import sudokuSolver
import copy
import os
from keras.models import load_model

# For tensorflow lent warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Loading detection model
model = load_model('includes/numbersDetection.h5')


# Read video function:
# Capture the video frame by frame for processing
# Change the video dimensions to square size 640x640
def read_video(video):
    success, frame = video.read()
    dim = (frame.shape[1] - frame.shape[0]) // 2
    frame_dim = frame[:, dim:dim + frame.shape[0]]
    frame = cv.resize(frame_dim, (900, 900))
    return frame, success


# Preprocess image function:
# Convert the image to grayscale
# Add some blur for easier detection
# Apply adaptive threshold
def image_preprocessing(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_blur = cv.GaussianBlur(image_gray, (5, 5), 2)
    image_threshold = cv.adaptiveThreshold(image_blur, 255, 1, 1, 11, 2)
    return image_threshold


# Find all contours function:
# We entered a threshold image and the function find all contours and return it
def find_all_contours(image_threshold):
    contours, hierarchy = cv.findContours(image_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


# Find biggest contour function:
# Extract the biggest contours because the sudoku board is the biggest one
# We loop in all contours
# Then check the area of each one because the small contours it's maybe noise or other unimportant details
# Finally get only biggest shapes points
def find_biggest_contour(contours):
    max_area = 0
    biggest_contour = None
    for contour in contours:
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            biggest_contour = contour
    return biggest_contour


# Get corners from contour function:
# This function return 4 corners if there is a board
# These 4 corners will be the corners of the sudoku board
def get_corners_from_contours(biggest_contour):
    corner_amount = 4
    max_iter = 200
    coefficient = 1
    while max_iter > 0 and coefficient >= 0:
        max_iter = max_iter - 1
        epsilon = coefficient * cv.arcLength(biggest_contour, True)
        poly_approx = cv.approxPolyDP(biggest_contour, epsilon, True)
        hull = cv.convexHull(poly_approx)
        if len(hull) == corner_amount:
            return hull
        else:
            if len(hull) > corner_amount:
                coefficient += .01
            else:
                coefficient -= .01
    return None


# Two matrices are equal function:
# Compare every single elements of two matrices and return if all items are equal
# Used for compare the solution between the current frame and the previous one
def two_matrices_are_equal(matrix_one, matrix_two):
    for row in range(9):
        for col in range(9):
            if matrix_one[row][col] != matrix_two[row][col]:
                return False
    return True


# Side lengths are too different function:
# This function is used as the first criteria for detecting whether
# The contour is a sudoku board or not: length of sides can't be too different (sudoku board is square)
# Return if the longest size is > the shortest size * eps_scale
def side_lengths_are_too_different(A, B, C, D, eps_scale):
    AB = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    AD = math.sqrt((A[0]-D[0])**2 + (A[1]-D[1])**2)
    BC = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
    CD = math.sqrt((C[0]-D[0])**2 + (C[1]-D[1])**2)
    shortest = min(AB, AD, BC, CD)
    longest = max(AB, AD, BC, CD)
    return longest > eps_scale * shortest


# approx 90 degrees function:
# This function is used as the second criteria for detecting whether
# All 4 angles of the sudoku board has to be approximately 90 degree
# Approximately 90 degrees with tolerance epsilon
def approx_90_degrees(angle, epsilon):
    return abs(angle - 90) < epsilon


# Angle between function:
# Return the angle between 2 vectors in degrees
def angle_between(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector2)
    angle = np.arccos(dot_product)
    angle = angle * 57.2958
    return angle


# Digit component function:
# This function is used for separating the digit from noise in each box of the boxes
# The sudoku board will be cropped into 9x9 small square image in the split boxes function
# each of those box is a cropped image
# Start from component 1 (not 0) because we want to leave out the background
def digit_component(image):
    image = image.astype(np.uint8)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    if len(sizes) <= 1:
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    output_image = np.zeros(output.shape)
    output_image.fill(255)
    output_image[output == max_label] = 0
    return output_image


# Get best shift function:
# Calculate how to centralize the image using its center of mass
def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty


# Shift function:
# Shift the image using what get best shift returns
def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv.warpAffine(img, M, (cols, rows))
    return shifted


# Reorder corners function:
# Every frame we get the corners in the different order so we get errors in the other steps
# So we change corners location to single format
# The top left one is the first corner and the bottom right is the last corner and so on
def reorder_corners(corners):
    board = np.zeros((4, 2), np.float32)
    corners = corners.reshape(4, 2)

    # Find top left corner ( the sum of coordinates is the smallest )
    sum = 10000
    index = 0
    for i in range(4):
        if corners[i][0] + corners[i][1] < sum:
            sum = corners[i][0] + corners[i][1]
            index = i
    board[0] = corners[index]
    corners = np.delete(corners, index, 0)

    # Find bottom right corner ( the sum of coordinates is the biggest )
    sum = 0
    for i in range(3):
        if corners[i][0] + corners[i][1] > sum:
            sum = corners[i][0] + corners[i][1]
            index = i
    board[2] = corners[index]
    corners = np.delete(corners, index, 0)

    # Find top right: only 2 points left so we check it and take the smallest is the top right
    # And the other one is bottom left corner
    if corners[0][0] > corners[1][0]:
        board[1] = corners[0]
        board[3] = corners[1]
    else:
        board[1] = corners[1]
        board[3] = corners[0]

    board = board.reshape(4, 2)
    A = board[0]
    B = board[1]
    C = board[2]
    D = board[3]
    return board, A, B, C, D


# Prepossessing for model function:
# After warp the perspective then the wrapped image contains only the chopped sudoku board
# So we do some image processing to get ready for recognizing digits model
def prepossessing_for_model(main_board):
    main_board = cv.cvtColor(main_board, cv.COLOR_BGR2GRAY)
    main_board = cv.GaussianBlur(main_board, (5, 5), 2)
    main_board = cv.adaptiveThreshold(main_board, 255, 1, 1, 11, 2)
    main_board = cv.bitwise_not(main_board)
    _, main_board = cv.threshold(main_board, 10, 255, cv.THRESH_BINARY)
    return main_board


# Prepare and normalize the image to get ready for digit recognition
# Reshape the image and flat it to one array
def prepare(img_array):
    new_array = img_array.reshape(-1, 28, 28, 1)
    new_array = new_array.astype(np.float32)
    new_array /= 255
    return new_array


# Get prediction function:
# First we init an empty grid to store the sudoku board digits
# Remove all boundaries on the edges of each image
# If this is a white cell, set grid[i][j] to 0 and continue on the next image
# Then clear the digits images and remove noise from it
# Resize images to 28x28 and apply binary threshold to get ready for recognition model
# recognise each image and return the array of grid numbers
def get_prediction(main_board):
    # Init empty 9 by 9 grid
    grid_dim = 9
    grid = []
    for i in range(grid_dim):
        row = []
        for j in range(grid_dim):
            row.append(0)
        grid.append(row)

    # Calculate the width and height to split main board to 81 image
    height = main_board.shape[0] // 9
    width = main_board.shape[1] // 9

    # Offset is used to get rid of the boundaries
    offset_width = math.floor(width / 10)
    offset_height = math.floor(height / 10)

    # Split the sudoku board into 9x9 squares ( 81 images )
    for i in range(grid_dim):
        for j in range(grid_dim):

            # Crop with offset ( we don't want to include the boundaries )
            crop_image = main_board[height * i + offset_height:height * (i + 1) - offset_height, width * j + offset_width:width * (j + 1) - offset_width]

            # But after that it will still have some boundary lines left
            # So we remove all black lines near the edges of each image
            # The ratio = 0.6 means if 60% of the pixels are black then remove this boundaries
            ratio = 0.6

            # Top
            while np.sum(crop_image[0]) <= (1 - ratio) * crop_image.shape[1] * 255:
                crop_image = crop_image[1:]

            # Bottom
            while np.sum(crop_image[:, -1]) <= (1 - ratio) * crop_image.shape[1] * 255:
                crop_image = np.delete(crop_image, -1, 1)

            # Left
            while np.sum(crop_image[:, 0]) <= (1 - ratio) * crop_image.shape[0] * 255:
                crop_image = np.delete(crop_image, 0, 1)

            # Right
            while np.sum(crop_image[-1]) <= (1 - ratio) * crop_image.shape[0] * 255:
                crop_image = crop_image[:-1]

            # Take the largest connected component ( the digit ) and remove all noises from it
            crop_image = cv.bitwise_not(crop_image)
            crop_image = digit_component(crop_image)

            # Resize each image to prepare it to the model
            digit_pic_size = 28
            crop_image = cv.resize(crop_image, (digit_pic_size, digit_pic_size))

            # If this is a white cell then set grid[i][j] to 0 because it's a blank cell then continue on the next image
            if crop_image.sum() >= digit_pic_size ** 2 * 255 - digit_pic_size * 1 * 255:
                grid[i][j] = 0
                continue

            # Detecting white cell if there is a huge white area in the center of the image
            center_width = crop_image.shape[1] // 2
            center_height = crop_image.shape[0] // 2
            x_start = center_height // 2
            x_end = center_height // 2 + center_height
            y_start = center_width // 2
            y_end = center_width // 2 + center_width
            center_region = crop_image[x_start:x_end, y_start:y_end]

            # So if we detect a white image ( blank square ) then we place it as zero in grid and continue
            if center_region.sum() >= center_width * center_height * 255 - 255:
                grid[i][j] = 0
                continue

            # After that we apply binary threshold to make digits more clear
            _, crop_image = cv.threshold(crop_image, 200, 255, cv.THRESH_BINARY)
            crop_image = crop_image.astype(np.uint8)

            # Centralize the image according to center of mass
            crop_image = cv.bitwise_not(crop_image)
            shift_x, shift_y = get_best_shift(crop_image)
            shifted = shift(crop_image, shift_x, shift_y)
            crop_image = shifted
            crop_image = cv.bitwise_not(crop_image)

            # Convert to prepare format to recognize ( the images ready for detection model )
            crop_image = prepare(crop_image)

            # Call detection model for each image and get it's detection then add it to the grid array
            prediction = model.predict([crop_image])
            grid[i][j] = np.argmax(prediction[0]) + 1
    return grid


# Write solution on image function:
# Get the image and its width and height then split it to 81 boxes ( 9x9 )
# Check the grid and if the grid not equal to zero ( this box not blank and have a number written on the grid ) then leave it
# Get the grid values after solve the game and convert the values to string to put it in the image
# Calculate the center of each box of 81 boxes of the grid to place the text on center and in the right size
# Calculate the font scale
# Finally put the text on each image on screen in the right place
def write_solution_on_image(image, grid, user_grid):
    grid_size = 9
    width = image.shape[1] // grid_size
    height = image.shape[0] // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            if user_grid[i][j] != 0:
                continue

            text = str(grid[i][j])
            offset_x = width // 15
            offset_y = height // 15
            (text_height, text_width), baseLine = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
            font_scale = 0.5 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width*j + math.floor((width - text_width) / 2) + offset_x
            bottom_left_corner_y = height*(i+1) - math.floor((height - text_height) / 2) + offset_y
            image = cv.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    return image


# Recognize and solve sudoku function:
# This function take an image from the camera and doing some image processing to find the sudoku board on this image
# After that we recognizing the digits in this board
# Then solve the sudoku puzzle and print the result back on the same image
# Finally return that image to show it to the user
def recognize_and_solve_sudoku(image, old_sudoku):
    # Step [1]: Preprocessing the image
    image_threshold = image_preprocessing(image)

    # Step [2]: Finding all contours
    contours = find_all_contours(image_threshold)

    # Step [3]: Finding the biggest contour because the biggest one is the game board
    biggest_contour = find_biggest_contour(contours)

    # If there is no sudoku in the scene then return the same input image
    if biggest_contour is None:
        return image

    # Step [4]: Get the corners of the sudoku board to use it in wrap perspective function
    corners = get_corners_from_contours(biggest_contour)

    # If there is no corners then no sudoku in the scene so we return the same input image
    if corners is None:
        return image

    # Step [5]: Reorder corners
    board, A, B, C, D = reorder_corners(corners)

    # After having found 4 corners A B C D then we check if ABCD is approximately square
    # Because the sudoku is always square board
    # First check the angles is 90 degree or not and if the angle is not 90 degree then return the input image
    # Because the sudoku angles is approximately 90
    AB = B - A
    AD = D - A
    BC = C - B
    DC = C - D
    eps_angle = 20
    if not (approx_90_degrees(angle_between(AB, AD), eps_angle)
            and approx_90_degrees(angle_between(AB, BC), eps_angle)
            and approx_90_degrees(angle_between(BC, DC), eps_angle)
            and approx_90_degrees(angle_between(AD, DC), eps_angle)):
        return image

    # The next step is to check the square board is the lengths should be approximately equal for AB, AD, BC, DC
    # Longest and shortest sides have to be approximately equal
    # But the longest side can't be longer than epsScale * shortest
    # So if they are longer than this value then we return the input image
    eps_scale = 1.2
    if side_lengths_are_too_different(A, B, C, D, eps_scale):
        return image

    # Now we are sure that this points A B C D correspond to 4 corners of a sudoku board
    # The width of the sudoku board
    (tl, tr, br, bl) = board
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # And the height of the sudoku board
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # The final dimensions
    max_width = max(int(width_A), int(width_B))
    max_height = max(int(height_A), int(height_B))

    # Construct the perspective destination points which will be used to map the screen main view
    screen = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], np.float32)

    # Step [6]: Get the game board from perspective
    cv.drawContours(image, corners, -1, (0, 255, 0), 15)
    transform_matrix = cv.getPerspectiveTransform(board, screen)
    main_board = cv.warpPerspective(image, transform_matrix, (max_width, max_height))
    original_board_wrap = np.copy(main_board)

    # Step [7]: Get the board digits recognition
    main_board = prepossessing_for_model(main_board)
    grid = get_prediction(main_board)
    user_grid = copy.deepcopy(grid)

    # Step [8]: Solve sudoku after recognizing each digits of the board
    # If the board is the same board from the last camera frame then print the same solution, no need to solve it again
    if (old_sudoku is not None) and two_matrices_are_equal(old_sudoku, grid):
        if sudokuSolver.all_board_non_zero(grid):
            original_board_wrap = write_solution_on_image(original_board_wrap, old_sudoku, user_grid)
    else:  # Else if there is a different board then solve it again
        sudokuSolver.solve_sudoku(grid)
        # If we got a solution from the previous step then write it on image
        if sudokuSolver.all_board_non_zero(grid):
            original_board_wrap = write_solution_on_image(original_board_wrap, grid, user_grid)
            old_sudoku = copy.deepcopy(grid)

    # Finally apply inverse perspective transformation and paste the solutions on top of the original image
    result_sudoku = cv.warpPerspective(original_board_wrap, transform_matrix, (image.shape[1], image.shape[0]), flags=cv.WARP_INVERSE_MAP)
    result = np.where(result_sudoku.sum(axis=-1, keepdims=True) != 0, result_sudoku, image)
    return result

