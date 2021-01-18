# Real Time Sudoku Solver
### Introduction
Sudoku is one of the most popular puzzle of all time. The goal of Sudoku is to fill (9x9) grid such that each row, and each column and the (3x3) small grid contains all of the digits between 1 to 9. And in this project I create a Sudoku solver which recognise the elements of Sudoku puzzle in the input video and providing a digital solution overlays on the top of the original image in real time (in live computer webcam) by using computer vision techniques to recognize and extract digits images from the grid and machine learning model (CNN) to get that digits detections. Sudoku Solver is the collection of very basic image processing techniques. A very good way to start is the OpenCV library which can be compiled on almost all the platforms.

### Computer Vision
Computer vision is the field that deals with how computers can be made for gaining high level understanding from digital images or videos. From the perspective of engineering, it seeks to automate tasks that the human visual system can do. Computer vision is concerned with the theory behind artificial systems that extract information from images. Computer vision is useful to emulate human vision like object recognition, defect detection or automatic driving etc.
<br />
<br />
# Knowledges required for this project
### Python programming language
Basic knowledge of coding and syntax formats of python is essential for this project. Python is advantageous as it easy to comprehend and consists of large number of inbult libraries that facilitate faster outputs. 
- Learn python through tutorials from [python.org](https://www.python.org/about/gettingstarted/)
- Download python from [python.org](https://www.python.org/downloads/)

### OpenCV library
Basic knowledge of opencv regarding use applications and syntax are essential here. It is a library in python that helps with computer vision and is more convenient as it provides large number of functions for conversions and processing. 
- Learn OpenCV from [opencv.org](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)
- Install OpenCV using pip ``` pip install opencv-python ```
- Install OpenCV in anaconda environment ``` conda install -c conda-forge opencv ```

### Numpy library
Basic knowledge of numpy including syntax and computational part are essentail to form image arrays. It is a lirary in pyton used to perform mathematical computation. 
- Learn Numpy from [numpy.org](http://www.numpy.org/)
- Install Numpy usin pip ``` pip install numpy ```
- Install Numpy in anaconda environment ``` conda install numpy ```

### Convolution Neural Network (CNN)
Convolution neural network is one of the deep learning algorithm that used to classify large image data into multi classes such as classify the images of dogs and cats to 2 classes or the image of other things to many classes. You shold have basic knowledge of deep learning and CNNs because I use it in the number detection and get the right class of each number. Or you can use another type of number recognition called OCR like (pytersseract) or any other OCR, and it will do the same work. 
- Learn CNN from [tensorflow.org](https://www.tensorflow.org/tutorials/images/cnn)
- Install Keras ``` pip install Keras ```
- Install Keras in anaconda environment ``` conda install -c conda-forge keras ```
<br />
<br />
# Usage
To run this project make sure you have the required installation of Python, OpenCV, Tensorflow and Keras then follow the steps given below:
- Clone or download this repository ``` https://github.com/ali-mohamed-nasser/Real-Time-Sudoku-Solver.git ```
- Run the file ``` main.py ```
- You don't need to train CNN on your own. I have trained a CNN model and saved the architecture in ``` numbersDetection.h5 ```
<br />
<br />
# How does it work?
First I read an input video from the webcam frame by frame and resize that video into square (900x900) video to work with it and after that I apply the following steps for each frame of that video.

### Image Processing
Convert the image to grayscale and apply some gaussian blur for easier edges detection then apply **image thresholding** which is the simplest method of image segmentation. From a grayscale image, thresholding can be used to create binary images. And there is many types of image thresholding:
- Simple thresholding
- Ostu's Binarization
- Adaptive Thresholding

I used the **Adaptive Thresholding** because this kind of thresholding calculate the threshold value for smaller regions and therefore, so there will be different threshold values for different regions. And this way give us better results in different lighting conditions becouse apply this threshold will remove unwanted noice.

<img src="https://github.com/ali-mohamed-nasser/Real-Time-Sudoku-Solver/blob/main/images/preproccessing.png" width="1200">

### Finding Cotours & getting the biggest one
Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.And for better accuracy, you should use binary images. So before finding contours I applied adaptive threshold or you can use Canny edge detection. Finally I extract the biggest contours because the sudoku board is the biggest one, then I wrap the perspective to extract the board.

<img src="https://github.com/ali-mohamed-nasser/Real-Time-Sudoku-Solver/blob/main/images/contours.png" width="1200">

*Note: In OpenCV, finding contours is like finding white object from black background. So remember, object to be found should be white and background should be black.*

### Getting digits from the grid
After getting the board in the last step I split that board to 81 image (each cell will be an image). And I clear that images one more time using thresholding to make the digits images more clear and sharp to pass it to the number detection model.

<img src="https://github.com/ali-mohamed-nasser/Real-Time-Sudoku-Solver/blob/main/images/split.png" width="1200">

### Recognising the value of digits
CNN is used for optocal character recognition here, and then it is further processed. The CNN convert the value of recognised digit to a string that can I used for further operations. The output of this opration is the number of each image and the empty cells will replaced by zero.

<img src="https://github.com/ali-mohamed-nasser/Real-Time-Sudoku-Solver/blob/main/images/numbers.png" width="900">

### Solve the sudoku board and display the solution
I use an external program that take an array of numbers (9x9) array. Then solve that array and return the result to place it on the image. To solve this array I use the backtracking algorithm which is a technique for solving problems recursively by trying to build a solution incrementally, one piece at a time, removing those solutions that fail to satisfy the constraints of the problem at any point. Finally overlay the corresponding calculated results on to live video.

Backtracking Algorithm     |  Overlay Solution
:-------------------------:|:-------------------------:
![](https://github.com/ali-mohamed-nasser/Real-Time-Sudoku-Solver/blob/main/images/backtraking.gif) | ![](https://github.com/ali-mohamed-nasser/Real-Time-Sudoku-Solver/blob/main/images/place.png)
<br />
<br />
# References
Here is a list of sources I used to build this project:
- Data used to train the CNN model is [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) for Computer Fonts, instead of MNIST for hand written digits.
- OpenCV Tutorial on [youtube.com](https://www.youtube.com/watch?v=qOXDoYUgNlU)
- How to improve Digit Recognition Accuracy [medium.com](https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4)
- [Find 4 corners from contour](https://www.programcreek.com/python/example/89417/cv2.arcLength)
- [How to extract Sudoku Grid](https://maker.pro/raspberry-pi/tutorial/grid-detection-with-opencv-on-raspberry-pi)
- OpenCV Tutorial from [opencv.org](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)
- And lot of ideas (and some codes) from [here](https://github.com/anhminhtran235/real_time_sudoku_solver)
