#importing necessary modules
import cv2 as cv
import cv2.aruco as aruco
import os
import numpy as np

#opening the image
img1 = cv.imread(r"C:\Users\ayush\Downloads\CVtask.png")
#resizing the image for better visibility
img = cv.resize(img1, (0, 0), fx = 0.6, fy = 0.6)

font = cv.FONT_HERSHEY_COMPLEX
blank = np.zeros(img.shape, dtype='uint8')

#Converting image to grey scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Finding countours 
canny = cv.Canny(gray, 0, 35)
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

#Iterating through contours
for contour in contours:
    #Find the squares in the image by going through the contours
    approx = cv.approxPolyDP(contour, 0.01* cv.arcLength(contour, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 4:
        x1 ,y1, w, h = cv.boundingRect(approx)
        aspectRatio = float(w)/h
        print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
          
          #Putting the values of co-ordinates of the vertices of each square detected
          n = approx.ravel()
          i = 0

          for j in n:
              if (i % 2 == 0):
                  x = n[i]
                  y = n[i + 1]

                  # String containing the co-ordinates.
                  string = str(x) + " " + str(y)

                  if (i == 0):
                      # text on topmost co-ordinate.
                      cv.putText(img, "Arrow tip", (x, y), font, 0.5, (255, 0, 0))
                  else:
                      # text on remaining co-ordinates.
                      cv.putText(img, string, (x, y), font, 0.5, (0, 255, 0))
              i = i + 1


cv.imshow("shapes", img)

#Defining the function to rotate a image
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result

#Defining the function to find the ID of Aruco Markers in image
def findaruco(img,draw=True):
    imggray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    arucoDict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    arucoParam= aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imggray,arucoDict, parameters=arucoParam)
    print(ids)

#Reading the first Aruco marker 'Ha'
img1 = cv.imread(r"C:\Users\ayush\Downloads\Ha.jpg")
findaruco(img1)
#As its ID is 3, we need to paste it on the Black square.
#So we need to rotate the image in such a way, so that black square become vertical.
#We will find out the angle of rotation and other details using the coordinates of squares found earlier.
rotate=rotate_image(img, 20)
#Resze the Aruco markers to the dimensions of target square
small_img = cv.resize(img1,(189,189))
#Pasting the Aruco on square
rotate[177:366,785:974] = small_img
#Rotating the the final image to its original direction
img=rotate_image(rotate, -20)

#Repeating the above process of pasting on other Aruco Markers.

img2 = cv.imread(r"C:\Users\ayush\Downloads\HaHa.jpg")
findaruco(img2)
rotate=rotate_image(img, -28)
small_img = cv.resize(img2,(325,325))
rotate[237:562,227:552] = small_img
img=rotate_image(rotate, +28)



img3 = cv.imread(r"C:\Users\ayush\Downloads\LMAO.jpg")
findaruco(img3)
rotate=rotate_image(img, 13)
small_img = cv.resize(img3,(245,245))
rotate[123:368,55:300] = small_img
img=rotate_image(rotate, -13)



img4= cv.imread(r"C:\Users\ayush\Downloads\XD.jpg")
findaruco(img4)
small_img = cv.resize(img4,(213,213))
img[43:256,701:914] = small_img

#Showing the final image
cv.imshow("img", img)

#Writing the final image
cv.imwrite("final.jpg",img)

cv.waitKey(0)
