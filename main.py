# Importing needed libraries and files
import cv2 as cv
import datetime
import includes.functions as func

# Load and set up the camera
# Get the video capture object for the camera
video = cv.VideoCapture(cv.CAP_DSHOW)

# Calculate frame rate
fps_start_time = datetime.datetime.now()
fps_number = 0
total_frames = 0

# Turn on the camera
old_sudoku = None
while True:
    frame, success = func.read_video(video)

    # Continue calculating the frame rate
    total_frames = total_frames + 1
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps_number = 0.0
    else:
        fps_number = (total_frames / time_diff.seconds)
    fps_text = "FPS: {:.1f}".format(fps_number)

    if success:
        # Recognize and solve sudoku board
        sudoku_game = func.recognize_and_solve_sudoku(frame, old_sudoku)

        # Print frames rate and showing image data
        cv.putText(sudoku_game, fps_text, (5, 20), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        cv.imshow("Real Time Sudoku Solver", sudoku_game)

        # Press [Q] to stop the camera and quit the app
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# After the loop release the capture object
# Then destroy all the windows
video.release()
cv.destroyAllWindows()