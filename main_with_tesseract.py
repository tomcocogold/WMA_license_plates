import cv2 as cv
import numpy as np
import pytesseract


def readText(image_):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    text_from_image = pytesseract.image_to_string(image_, config=custom_config)
    return text_from_image


img_number = 2
path = str(img_number) + '.jpg'
image = cv.imread(path)
img = cv.resize(image, (1156, 652), interpolation=cv.INTER_CUBIC)

print('This is image number {}'.format(img_number))
cv.imshow("Photo", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Scale', gray)

filtered = cv.bilateralFilter(gray, 5, 100, 100)
cv.imshow('Bilateral Filter', filtered)

canny = cv.Canny(filtered, 50, 150, 0, 3, True)
cv.imshow('Canny Edges', canny)

contours, hierarchies = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print('{} contour(s) found!'.format(len(contours)))
# print(hierarchies)
# we create the list of hull contours
hull = []
for i in range(len(contours)):
    hull.append(cv.convexHull(contours[i], False))

img_copy = img.copy()
cv.drawContours(img_copy, contours, -1, (0, 255, 0), 1)
cv.imshow('Contours', img_copy)

img_copy = img.copy()
cv.drawContours(img_copy, hull, -1, (0, 255, 0), 1)
cv.imshow('Convex Hull Contours', img_copy)

license_plates = list()
img_copy = img.copy()
for hull_contour in hull:
    # bounding rectangle
    x, y, w, h = cv.boundingRect(hull_contour)
    aspect_ratio = float(w) / h
    rect_area = float(w) * h

    # convexHull
    hull_contour = cv.convexHull(hull_contour)
    hull_area = cv.contourArea(hull_contour)
    hull_extent = float(hull_area) / rect_area
    hull_perimeter = cv.arcLength(hull_contour, True)
    cropped = img[y:(y + h), x:(x + w)]
    cropped_gray = gray[y:(y + h), x:(x + w)]
    # approx_shape = cv.approxPolyDP(hull_contour, 0.05 * cv.arcLength(hull_contour, True), True)
    # the above function isn't working too well

    if 2.5 < aspect_ratio < 5.5 and hull_extent > 0.6 and cv.mean(cropped_gray)[0] > 70 and (
            5080 < hull_area < 5085 or 530 < hull_area < 540 or 2040 < hull_area < 2050 or 1460 < hull_area < 1470
            or 1560 < hull_area < 1570 or 8960 < hull_area < 8970 or 3800 < hull_area < 3810 or 16610 < hull_area < 16620
            or 5850 < hull_area < 5860 or 581 < hull_area < 590 or 6700 < hull_area < 6710 or 4750 < hull_area < 4760
            or 8010 < hull_area < 8020 or 5000 < hull_area < 5010 or 7200 < hull_area < 7210 or 2910 < hull_area < 2920
            or 2100 < hull_area < 2110 or 11900 < hull_area < 11910 or 20950 < hull_area < 20960 or 9350 < hull_area < 9360
            or 4160 < hull_area < 4170 or 12447 < hull_area < 12457 or 6100 < hull_area < 6110 or 3450 < hull_area < 3460):
        license_plates.append(hull_contour)
        cv.drawContours(img_copy, [hull_contour], 0, (0, 255, 0), 2)

cv.imshow("License Plates, image " + str(img_number), img_copy)
print('There is(are) {} license plate(s) in this image'.format(len(license_plates)))

index = 0
for license_plate in license_plates:

    cv.waitKey(0)
    index = index + 1
    x, y, w, h = cv.boundingRect(license_plate)
    cropped = img[y:(y + h), x:(x + w)]
    # when we use numpy we have to switch the coordinates x and y because there is a different coordinate system
    cropped_resized = cv.resize(cropped, (520, 114), interpolation=cv.INTER_CUBIC)
    cv.imshow("Cropped License Plate " + str(index), cropped_resized)
    gray = cv.cvtColor(cropped_resized, cv.COLOR_BGR2GRAY)
    cv.imshow("Cropped Grayscale License Plate " + str(index), gray)
    if cv.contourArea(license_plate) < 600:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        filtered = cv.filter2D(src=gray, ddepth=-1, kernel=kernel)
        cv.imshow('Sharp License Plate' + str(index), filtered)
    else:
        filtered = cv.bilateralFilter(gray, 5, 100, 100)
        cv.imshow("Cropped Filtered License Plate " + str(index), filtered)

    # make a function that chooses the threshold value based on the mean intensity of the filtered picture
    mean_val = cv.mean(filtered)
    # print('mean value of the grayscale image is :' + str(mean_val))
    if 255 <= mean_val[0] < 200:
        threshold = cv.adaptiveThreshold(filtered, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 155, 50)
    elif 150 < mean_val[0] <= 200:
        threshold = cv.adaptiveThreshold(filtered, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 155, -30)
    elif 100 < mean_val[0] <= 150:
        threshold = cv.adaptiveThreshold(filtered, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 155, 0)
    elif 50 < mean_val[0] <= 100:
        threshold = cv.adaptiveThreshold(filtered, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 155, +10)
    else:
        threshold = cv.adaptiveThreshold(filtered, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 155, -10)

    cv.imshow('Cropped Threshold License Plate ' + str(index), threshold)
    canny = cv.Canny(threshold, 50, 150, 0, 3, True)
    cv.imshow("Cropped Canny License Plate " + str(index), canny)
    contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cropped_resized_copy = cropped_resized.copy()
    cv.drawContours(cropped_resized_copy, contours, -1, (0, 255, 0), 1)
    cv.imshow("Cropped with contours License Plate " + str(index), cropped_resized_copy)
    letters = []
    cropped_resized_copy = cropped_resized.copy()
    for letter in contours:
        if cv.contourArea(letter) > 250:
            letters.append(letter)
            cv.drawContours(cropped_resized_copy, [letter], 0, (0, 255, 0), 2)
    cv.imshow("Cropped License Plate with letters", cropped_resized_copy)

    print("License Plate Number " + str(index) + ": ", end=" ")
    print(readText(cropped), end="")

cv.waitKey(0)
cv.destroyAllWindows()
