# Lane-Detection

## **PROJECT DESCRIPTION**

The aim of this project is to do simple Lane Detection to mimic Lane Departure Warning systems used in Self Driving Cars. The task is to design an algorithm to detect lanes on the road, as well as estimate the road curvature to predict car turns.

Please refer to [Project Report](https://github.com/sanhuezapablo/Lane-Detection/blob/master/Report/Project%202.pdf) for further description

### Preparing the Input

<p align="center">
  <img src="/Images/combined_hsl.png" alt="Input Prep">
</p>

- The image is undistorted using the camera parameters provided
- The image is denoised
- The color scale is changed from RGB to HSL
- The region of interest (ROI) is extracted. I haveonly cropped out the top half of the image (sky) as it cannot be assumed that the car does not change lanes, and hence define small regions around lanes as ROIs

<p align="center">
  <img src="/Images/homography.png" alt="Homography">
</p>

### Lane Detection Candidates

Here two approaches are used 1) Hough Lines and 2) Histogram of Lane Pixels

#### Hough Lines

<p align="center">
  <img src="/Images/hough.gif" alt="Detect Tag">
</p>

- Hough lines from the edge image acquired earlier are found
- The peak Hough lines are found and extrapolated

This approach however doesnt successfully account for Curved Roads 

#### Histogram of Lane Pixels

<p align="center">
  <img src="/Images/histogram.png" alt="histo">
</p>

- A histogram of pixel count along the y-axis is generated
- The top regions with highest pixel count are extracted as they correspond to lanes

### Refining Lane Detection and Turn Prediction

<p align="center">
  <img src="/Images/histo.gif" alt="final_histo">
</p>

A polynomial is fit to the detected lane candidates for better results.
A polygon is fit between the two detected anes.

I have used the slope method to implement Turn Prediction.

## **DEPENDANCIES**

- Python 3
- OpenCV
- Numpy
- Copy (built-in)


## **FILE DESCRIPTION**

- Code Folder/[Code 1(Both).py](https://github.com/sanhuezapablo/Lane-Detection/blob/master/Code/Code%201(Both).py) - The code file can be run on both Project video and Challenge Video but isnt optimal for either
- Code Folder/[Code 2(Project only).py](https://github.com/sanhuezapablo/Lane-Detection/blob/master/Code/Code%202%20(Project%20Only).py) - The code file has been optimized for the Project Video
- Code Folder/[OPTIONAL_HoughLines_Project2.py](https://github.com/sanhuezapablo/Lane-Detection/blob/master/Code/OPTIONAL_HoughLines_Project2.py) - The code file where Hough Lines was attempted. It was later abandoned due to reasons mentioned in [Project Report](https://github.com/adheeshc/Lane-Detection/blob/master/Report/Project%202.pdf) 

- Datasets folder - Contains 2 video input files 

- Images folder - Contains images for github use (can be ignored)

- Output folder - Contains output videos

- References folder - Contains supplementary documents that aid in understanding

- Report folder - Contains [Project Report](https://github.com/sanhuezapablo/Lane-Detection/blob/master/Report/Project%202.pdf)

## **RUN INSTRUCTIONS**

- Make sure all dependancies are met
- Ensure the location of the input video files are correct in the code you're running
- Comment/Uncomment as reqd

- RUN Code 1(Both).py for both project and challenge video
- RUN Code 2(Project Only).py for project video (Optimized)
- RUN OPTIONAL_HoughLines_Project2.py for Hough Lines method
