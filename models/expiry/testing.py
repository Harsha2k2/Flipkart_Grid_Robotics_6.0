from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os

test_image = 'C:/Users/Harsha/Downloads/archive/dataset/test_images.txt'
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
	help="path to input image/text file of image filenames", default=test_image)
# args = vars(ap.parse_args())
args = ap.parse_args(args=[])
print(args.input)
# single input image
filetype = mimetypes.guess_type(args.input)[0]
print(filetype)
imagePaths = [args.input]
# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:
	# load the filenames in our testing file and initialize our list
	# of image paths
	filenames = open(args.input).read().strip().split("\n")
	imagePaths = []
	# loop over the filenames
	for f in filenames:
		# construct the full path to the image filename and then
		# update our image paths list
		p = os.path.sep.join([IMAGES_PATH, f])
		imagePaths.append(p)
print(imagePaths)

print("[INFO] loading object detector...")
model = load_model(MODEL_PATH)
# loop over the images that we'll be testing using our bounding box
# regression model
for imagePath in imagePaths:
	# load the input image (in Keras format) from disk and preprocess
	# it, scaling the pixel intensities to the range [0, 1]
  image = cv2.imread(imagePath)
  (h, w) = image.shape[:2]
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.resize(image,(224, 224))
  # image = cv2.erode(image, None, iterations=1)
  image = img_to_array(image) / 255.0
  image = np.expand_dims(image, axis=0)
  # make bounding box predictions on the input image
  preds = model.predict(image)[0]
  (startX, startY, endX, endY) = preds
  # load the input image (in OpenCV format), resize it such that it
  # fits on our screen, and grab its dimensions
  image = cv2.imread(imagePath)
  image = imutils.resize(image, width=600)
  (h, w) = image.shape[:2]
  print(h,w)
  # scale the predicted bounding box coordinates based on the image
  # dimensions
  startX = int(startX * w)
  startY = int(startY * h)
  endX = int(endX * w)
  endY = int(endY * h)
  # show the output image
  cv2_imshow(image)
  # show the output image
  print(image.shape)
  bb = image[startY-10:endY+10, startX-10: endX+10]
  # cv2.imwrite('/content/output/test.png', bb)
  cv2_imshow(bb)


  
