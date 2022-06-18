import cv2 as cv
import cv2.aruco as aruco
import os
import numpy as np

img1 = cv.imread(r"C:\Users\ayush\Downloads\CVtask.png")
font = cv.FONT_HERSHEY_COMPLEX
img = cv.resize(img1, (0, 0), fx = 0.6, fy = 0.6)


blank = np.zeros(img.shape, dtype='uint8')


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

canny = cv.Canny(gray, 0, 35)


ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)


contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')



for contour in contours:
    approx = cv.approxPolyDP(contour, 0.01* cv.arcLength(contour, True), True)
    #cv.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5

    if len(approx) == 4:
        x1 ,y1, w, h = cv.boundingRect(approx)
        aspectRatio = float(w)/h
        print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
          #cv.putText(img, "square", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
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

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result

def findaruco(img,draw=True):
    imggray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    arucoDict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    arucoParam= aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imggray,arucoDict, parameters=arucoParam)
    print(ids)
img1 = cv.imread(r"C:\Users\ayush\Downloads\Ha.jpg")
findaruco(img1)

rotate=rotate_image(img, 20)
small_img = cv.resize(img1,(189,189))
rotate[177:366,785:974] = small_img

img=rotate_image(rotate, -20)


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
cv.imshow("img", img)


cv.imwrite("final.jpg",img)

cv.waitKey(0)