<strong><em>Let's stitch together some images.</em></strong>

The code to do the image stitching originally came from Akash Kumar Singh (ksakash).  I have been modifying it to make it more understandable to myself.

<em>Modifications include</em>
<ul>
 	<li>Breaking up large functions into several smaller ones</li>
 	<li>Renaming variables with more meaningful names</li>
 	<li>Adding descriptive comments</li>
 	<li>Adding more output functions to see matrix values and images at intermediate steps during algorithm execution</li>
</ul>

<strong><em>Shout Out to the Creators of the SURF algorithm</em></strong>

SURF stands for Speeded Up Robust Features.  It is an algorithm to find keypoints and descriptors for features in an image.  A feature is, for example, an apple and the table it is on.  SURF was first presented in 2006 by Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool.

[SURF](https://people.ee.ethz.ch/~surf/index.html)

<em><strong>Description of the Image Stitching Algortihm</strong></em>

<strong>1</strong> getTransformationToOriginPlane()<br>

We are stitching together 2D images of our 3D world. Each image was taken by a camera that had a certain orientation within our 3D world. We first align all the images to one camera orientation.

We use the yaw-pitch-roll terminology of aeronautics to describe the 3 angles that rotated our camera away from its origin plane. We compute our rotation unwinding matrix and apply it to each image given their unique values for yaw, pitch, and roll.

<strong>2</strong> combinePair()

Image stitching is an iterative process. We start by stitching two images together.  Then we stitch the third to the previously stitched first and second images.  And so on.

<strong>2.1</strong> combinePair.getKeypoints()

Using the SURF algorithm we find the keypoints with their associated descriptors in each image.

<strong>2.2</strong> combinePair.matchKeypointDescriptors()

We then match up the keypoint descriptors from the two images.  This matching determines the seam along which we will stitch the two images.

<strong>2.3</strong> combinePair.getAffineTransformation()

Using the matched keypoint descriptors we can now compute the affine transformation matrix that will take image 2 and put in the place it needs to be inside image 1.

<strong>2.3.1</strong> combinePair() -&gt; warpPerspective image 1

Puts image 1 with correct scale and position inside the bounding rectangle shared by the two images.

<strong>2.3.2</strong> combinePair() -&gt; warpPerspective image 2

Puts image 2 with correct scale but not correct position inside the bounding rectangle shared by the two images.

<strong>2.3.3</strong> combinePair() -&gt; warpAffine image 2

Puts image 2 at the correct position inside the bounding rectangle shared by the two images.

<strong>2.3.4</strong> combinePair() -&gt; create mask of image 2

This mask matches the boundaries of image 2 and will be used to prepare image 2 for insertion into image 1.

<strong>2.3.5</strong> combinePair() -&gt; multiply image 2 by the mask

Prepare image 2 for insertion.

<strong>2.3.6</strong> combinePair() -&gt; create difference mask from mask of image 2

This difference mask determines the portion of image 1 that we must carve out to have a place for image 2 to be inserted.

<strong>2.3.7</strong> combinePair() -&gt; multiply image 1 by the difference mask

Carve out our place in image 1 in which we will insert image 2.

<strong>2.3.8</strong> combinePair() -&gt; add our two masked images

We add the two masked images and we are done. The two images have been combined!

&nbsp;

<em><strong>Input Images</strong></em>

Five different Aerial views of Fishermans Wharf in San Francisco.

<em>Image A</em>
<img src="https://github.com/TallDave67/cuda_stitch/blob/master/datasets/1/input/1.png" />

<em>Image B</em>
<img src="https://github.com/TallDave67/cuda_stitch/blob/master/datasets/1/input/2.png" />

<em>Image C</em>
<img src="https://github.com/TallDave67/cuda_stitch/blob/master/datasets/1/input/3.png" />

<em>Image D</em>
<img src="https://github.com/TallDave67/cuda_stitch/blob/master/datasets/1/input/4.png" />

<em>Image E</em>
<img src="https://github.com/TallDave67/cuda_stitch/blob/master/datasets/1/input/5.png" />

&nbsp;

<em><strong>Intermediate Images</strong></em>

Select images during algorithm execution where image C was being combined with the previous combination of image A and image B.

<em>Step 2.3.1: warpPerspective image AB</em>
<img src="https://cuberanger.com/wp-content/uploads/2021/08/2_3_1_warpedPerspectiveImgAB_gpu.png" />

<em>Step 2.3.2: warpPerspective image C</em>
<img src="https://cuberanger.com/wp-content/uploads/2021/08/2_3_2_warpedPerspectiveImgC_gpu.png" />

<em>Step 2.3.3: warpAffine image C</em>
<img src="https://cuberanger.com/wp-content/uploads/2021/08/2_3_3_warpedPerspectiveAffineImgC_gpu.png" />

<em>Step 2.3.4: mask for image C</em>
<img src="https://cuberanger.com/wp-content/uploads/2021/08/2_3_4_warpedPerspectiveAffineImgC_mask_gpu.png" />

<em>Step 2.3.7: difference mask multiplied by image AB</em>
<img src="https://cuberanger.com/wp-content/uploads/2021/08/2_3_7_warpedPerspectiveImgAB_gpu_multiply_by_difference_mask_gpu.png" />

<em>Step 2.3.8: combined image ABC</em>
<img src="https://cuberanger.com/wp-content/uploads/2021/08/2_3_8_combined_gpu.png" />

&nbsp;

<em><strong>Output Image</strong></em>

After 4 iterations the 5 images are combined into the final result.

<em>Output image ABCDE</em>
<img src="https://cuberanger.com/wp-content/uploads/2021/08/result.png" />
