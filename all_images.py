import cv2 as cv
import numpy as np

for i in range(15):
    img_number = i + 1
    path = str(img_number) + '.jpg'
    image = cv.imread(path)
    img = cv.resize(image, (1156, 652), interpolation=cv.INTER_CUBIC)

    print('This is image number {}'.format(img_number))
    #cv.imshow("Photo", img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow('Gray Scale', gray)

    filtered = cv.bilateralFilter(gray, 5, 100, 100)
    #cv.imshow('Bilateral Filter', filtered)

    canny = cv.Canny(filtered, 50, 150, 0, 3, True)
    #cv.imshow('Canny Edges', canny)

    contours, hierarchies = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print('{} contour(s) found!'.format(len(contours)))

    # we create the list of hull contours
    hull = []
    for i in range(len(contours)):
        hull.append(cv.convexHull(contours[i], False))

    img_copy = img.copy()
    cv.drawContours(img_copy, contours, -1, (0, 255, 0), 1)
    #cv.imshow('Contours', img_copy)

    license_plates = list()
    img_copy = img.copy()
    for contour in contours:
        # bounding rectangle
        # minAreaRect
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = float(w) / h

        perimeter = cv.arcLength(contour, True)
        area = cv.contourArea(contour)
        rect_area = float(w) * h
        extent = float(area) / rect_area

        # convexHull
        hull_contour = cv.convexHull(contour)
        hull_area = cv.contourArea(hull_contour)
        hull_extent = float(hull_area) / rect_area
        hull_perimeter = cv.arcLength(hull_contour, True)

        # approx_shape = cv.approxPolyDP(hull_contour, 0.5 * cv.arcLength(hull_contour, True), True)

        if 2.5 < aspect_ratio < 5.5 and hull_extent > 0.6 and 500 < hull_area < 25000:
            license_plates.append(hull_contour)
            cv.drawContours(img_copy, [hull_contour], 0, (0, 255, 0), 2)

    cv.imshow("License Plates, image " + str(img_number), img_copy)
    print('There is(are) {} license plate(s) in this image'.format(len(license_plates)))

cv.waitKey(0)
cv.destroyAllWindows()