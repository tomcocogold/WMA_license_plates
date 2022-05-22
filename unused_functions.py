import numpy as np
import cv2 as cv

# threshold = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,155,-50)
# cv.imshow('Threshold', threshold)

# kernel =cv.getStructuringElement(cv.MORPH_RECT,(3,3))
# morphed = cv.morphologyEx(threshold, cv.MORPH_CLOSE, (3,3),iterations=3)
# dilate = cv.dilate(threshold,np.ones((3,3),np.uint8), iterations=1)
# erode = cv.erode(dilate, np.ones((3,3),np.uint8), iterations=1)
# cv.imshow('Morphed', erode)

# kernel = np.array([[0, -1, 0],
#                 [-1, 5, -1],
#                [0, -1, 0]])
# sharp = cv.filter2D(src=img, ddepth=-1, kernel=kernel)
# cv.imshow('Sharp', sharp)

# for i in range(len(contours)):
#    cv.drawContours(img_copy, hull, i, (0, 255, 0), 1)

for cnt in license_plates:
    mask = np.zeros(gray.shape, np.uint8)
    cv.drawContours(mask, [cnt], 0, 255, -1)
    pixelpoints = np.transpose(np.nonzero(mask))
    cv.imshow("Maska", mask)

    #pixelpoints = cv.findNonZero(mask)
    #cv.imshow("pixelpoints", pixelpoints)

    leftmost = tuple(cnt[cnt[:, :,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    print("left " + str(leftmost), "right " + str(rightmost), "top " + str(topmost), "bottom " + str(bottommost))
    # wycinamy kontur
    cropped = img[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]
    cv.imshow("Cropped", cropped)
    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    cv.imshow("Cropped Grayscale", gray)
    filtered = cv.bilateralFilter(gray, 5, 200, 200)
    cv.imshow("Cropped Filtered", filtered)
    threshold = cv.adaptiveThreshold(filtered,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,155,-50)
    cv.imshow('Cropped Threshold', threshold)
    canny = cv.Canny(threshold, 50, 150, 0, 3, True)
    cv.imshow("Cropped Canny", canny)
    contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cv.drawContours(cropped,contours, -1, (0, 255, 0), 1)
    resized = cv.resize(cropped, (520, 114), interpolation=cv.INTER_CUBIC)
    cv.imshow("Cropped with contours", resized)
    cv.waitKey(0)

#pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv.findNonZero(mask)
    #cv.imshow("pixelpoints", pixelpoints)
    #leftmost = tuple(cnt[cnt[:, :,0].argmin()][0])
    #rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    #topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    #bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

# print("left " + str(leftmost), "right " + str(rightmost), "top " + str(topmost), "bottom " + str(bottommost))
# wycinamy kontur
# cropped = img[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]

 #(x, y) = np.where(mask == 255)
    #(topx, topy) = (np.min(x), np.min(y))
    #(bottomx, bottomy) = (np.max(x), np.max(y))
    #cropped = img[topx:bottomx, topy:bottomy]

    mask = np.zeros(gray.shape, np.uint8)
    cv.drawContours(mask, [license_plate], 0, 255, -1)
    cv.imshow("Maska", mask)

    and cv.mean(gray)[0] > 100
    and cv.mean(cropped)[1] < 150
    and cv.mean(cropped)[2] < 150 and cv.mean(cropped)[3] < 150

    1350 < hull_area < 21000