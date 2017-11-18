## Advanced Lane Finding Pipeline

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## 1. Calibrate Camera

The following code allows us to recover the calibration matrix of the camera. We only need to run it once at the beginning, since the calibration is a static feature of the lens.
We are using the 20 chessboard images that we were provided by Udacity. 

Of these 20 images, we noticed that on three of them we can not find the board. Inspecting them (images 1, 4 and 5) we notice that en each of them at least one of the chessboard corners is cropped. If we really wanted to use them, we could change the size of the chessboard to find, but instead, what we will do is use them as tests for our calibration.

<img src="examples/Calibration.png" width="480" alt="Combined Image" />

From the images above, we can immediately see the improvement we get after undistorting the original pictures. In particular, the lines look a lot more like straight lines once we apply the transformation. We are now ready for the next step

## 2. Obtain edges

In this step, we want to process the images to obtain a binary where the lanes are easy to identify and there is little else to distract away from them. 

To do so, we will use a few different techniques to isolate the pixels that are likely to define the lanes.
We will use:
* A greyscale version of the image.
* The S channel of the HLS colour representation.
* A threshold version of the HSV image.

The idea is that combining these different transformations, our lane identification will be more robust, since at least one of them might pick up the lanes.

All the code we use for going from the undistorted image to the bird's eye view of the edges can be found in Source/img_processing.py.

To pick up the likely lane pixels, we use a few different techniques. First we obtain different single channel versions of the image:
* Greyscale
* S channel from HLS
* L channel from HLS
We also obtain a fourth image using a hard coded thresholding on the HSV colour map that picks yellow and white.

After that, we applied first and second degree operators to the singel channel images (Sobel and Laplacian), as well as thresholding to the resulting images to obtain binaries.

<img src="examples/Edges.png" width="480" alt="Combined Image" />

We experimented with different combinations of the above, and settled on a Laplacian of the S channel, the Sobel on the greyscale image, and binaries for the S channel and the HSV yellow and white. We combined all of them to obtain the set of likely pixels of the lanes.

## 3. Perspective transformation

We want to change the perspective of our images to obtain a bird's eye view of the road, since this will allow us to calculate the curvature of the road, and the position of the car with respect to the centre of the lane.
To do so, we will use one of the straight lane images available to us, and play around with the parameters of the transformation until we obtain a satisfactory result

<img src="examples/Birds_eye.png" width="480" alt="Combined Image" />

## 4. Fitting a (lane) curve.

After obtaining the Bird's eye view of the likely lane pixels, it is time to try and obtain a more useful representation of the lanes. In the first project of this Nano-degree, we simply fitted a straight line. This time we will do something slightly more sophisticated.
First, we will try and identify where is the likely starting point each lane. To do so, we want to focus on the lower third of the image, and find where are the pixels concentrated.
Take the histogram of the x-coordinate of the pixels present in the lower third of the image.

<img src="examples/Histogram.png" width="480" alt="Combined Image" />

As we can see above, we are likely to identify the general location of the lane by looking at the modes of the above distribution, namely, the left hand side and right hand side modes are good starting points to look for the lanes.
Using this information, we do a sliding window search to find all the pixels that are likely to belong to the lane lines.
We draw a window around each of our initial points, and identify all the pixels that are within the window (which we colour red for the left lane and blue for the right).
The next window is drawn immediately above, and can slide left or righ if the number of pixels found is big enough, in which case the centre of the new window is taken to be the average x coordinate in the previous one.

<img src="examples/Window.png" width="480" alt="Combined Image" />

Having collected all the right hand lane and left hand lane pixels, we can now fit a curve to each of this sets, where we take y to be a function of x (because the lanes are close to vertical). The picture above illustrates both the sliding windows and th fitted curves.
Once we have identified the lanes in a frame, we expect that they will be in a similar place in the next one, and thus it is unnecessary to go through the full sliding window process again. Instead, we use the lanes (fitted curves) from the previous frame, and draw a search region around them. These regions are simply definied by applying a margin to the x-coordinate of each curve. We can then again identify all the pixels that fall under these regions, and fit new lanes for them.

<img src="examples/Mask.png" width="480" alt="Combined Image" />

## 5. Test on the video.

We have developed the tools to transform the image into a bird's eye view of the likely lane pixels, fitted the lane curves and drawn back the lane onto the original image. It looks like we are ready to have a go at the video.

<img src="examples/Lane.png" width="480" alt="Combined Image" />
