import cv2 as cv
import numpy as np
from letters_and_digits import letters_and_digits as lad
from wcontour import w_contour as w

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
print('{} contour(s) found!'.format(len(contours)))
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
    # ret, threshold = cv.threshold(filtered, 50, 255, cv.THRESH_BINARY_INV)
    cv.imshow('Cropped Threshold License Plate ' + str(index), threshold)
    canny = cv.Canny(threshold, 50, 150, 0, 3, True)
    cv.imshow("Cropped Canny License Plate " + str(index), canny)
    contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cropped_resized_copy = cropped_resized.copy()
    cv.drawContours(cropped_resized_copy, contours, -1, (0, 255, 0), 1)
    # we now change the size of found license_plates for easier comparison and digits and letters recognition
    # resized = cv.resize(cropped, (520, 114), interpolation=cv.INTER_CUBIC)
    cv.imshow("Cropped with contours License Plate " + str(index), cropped_resized_copy)
    letters = []
    cropped_resized_copy = cropped_resized.copy()
    for letter in contours:
        if cv.contourArea(letter) > 250:
            letters.append(letter)
            cv.drawContours(cropped_resized_copy, [letter], 0, (0, 255, 0), 2)
    cv.imshow("Cropped License Plate with letters", cropped_resized_copy)

    bounding_rectangles = []
    for letter in letters:
        bounding_rectangles.append(cv.boundingRect(letter)[0])
    coordinates = []
    for i in range(len(letters)):
        coordinates.append((letters[i], bounding_rectangles[i]))

    def sort_tuple_list(tup):
        tup.sort(key=lambda x: x[1])
        return tup
    # below there is a list of tuples where the first element is the letter contour and the list is sorted by the
    # x coordinate ascending (so we can print the letters from left to right)
    sorted_coordinates = sort_tuple_list(coordinates)

    detected_letters = []

    for letter in sorted_coordinates:
        compared_letters = [cv.matchShapes(letter[0], lad[84], 1, 0.0),#A (0)
                            cv.matchShapes(letter[0], lad[83], 1, 0.0),#B (1)
                            cv.matchShapes(letter[0], lad[82], 1, 0.0),#C (2)
                            cv.matchShapes(letter[0], lad[81], 1, 0.0),#D (3)
                            cv.matchShapes(letter[0], lad[80], 1, 0.0),#E (4)
                            cv.matchShapes(letter[0], lad[79], 1, 0.0),#F (5)
                            cv.matchShapes(letter[0], lad[78], 1, 0.0),#G (6)
                            cv.matchShapes(letter[0], lad[77], 1, 0.0),#H (7)
                            cv.matchShapes(letter[0], lad[76], 1, 0.0),#I (8)
                            cv.matchShapes(letter[0], lad[53], 1, 0.0),#J (9)
                            cv.matchShapes(letter[0], lad[45], 1, 0.0),#K (10)
                            cv.matchShapes(letter[0], lad[44], 1, 0.0),#L (11)
                            cv.matchShapes(letter[0], lad[50], 1, 0.0),#M (12)
                            cv.matchShapes(letter[0], lad[49], 1, 0.0),#N (13)
                            cv.matchShapes(letter[0], lad[43], 1, 0.0),#O (14)
                            cv.matchShapes(letter[0], lad[39], 1, 0.0),#P (15)
                            cv.matchShapes(letter[0], lad[41], 1, 0.0),#R (16)
                            cv.matchShapes(letter[0], lad[40], 1, 0.0),#S (17)
                            cv.matchShapes(letter[0], lad[63], 1, 0.0),#T (18)
                            cv.matchShapes(letter[0], lad[62], 1, 0.0),#U (19)
                            cv.matchShapes(letter[0], w, 1, 0.0),   #W (20)
                            cv.matchShapes(letter[0], lad[61], 1, 0.0),#X (21)
                            cv.matchShapes(letter[0], lad[69], 1, 0.0),#V (22)
                            cv.matchShapes(letter[0], lad[60], 1, 0.0),#Y (23)
                            cv.matchShapes(letter[0], lad[59], 1, 0.0),#Z (24)
                            cv.matchShapes(letter[0], lad[13], 1, 0.0),#0 (25)
                            cv.matchShapes(letter[0], lad[22], 1, 0.0),#1 (26)
                            cv.matchShapes(letter[0], lad[27], 1, 0.0),#2 (27)
                            cv.matchShapes(letter[0], lad[26], 1, 0.0),#3 (28)
                            cv.matchShapes(letter[0], lad[31], 1, 0.0),#4 (29)
                            cv.matchShapes(letter[0], lad[17], 1, 0.0),#5 (30)
                            cv.matchShapes(letter[0], lad[16], 1, 0.0),#6 (31)
                            cv.matchShapes(letter[0], lad[15], 1, 0.0),#7 (32)
                            cv.matchShapes(letter[0], lad[18], 1, 0.0),#8 (33)
                            cv.matchShapes(letter[0], lad[25], 1, 0.0)]#9 (34)
        # print(compared_letters)
        min_value = compared_letters[0]
        idx = 0
        for i in range(len(compared_letters)):
            if compared_letters[i] < min_value:
                min_value = compared_letters[i]
                idx = i
        # print(min_value, idx)
        if min_value > 0.9:
            pass
        else:
            if idx == 0:
                detected_letters.append('A')
            elif idx == 1:
                detected_letters.append('B')
            elif idx == 2:
                detected_letters.append('C')
            elif idx == 3:
                detected_letters.append('D')
            elif idx == 4:
                detected_letters.append('E')
            elif idx == 5:
                detected_letters.append('F')
            elif idx == 6:
                detected_letters.append('G')
            elif idx == 7:
                detected_letters.append('H')
            elif idx == 8:
                detected_letters.append('I')
            elif idx == 9:
                detected_letters.append('J')
            elif idx == 10:
                detected_letters.append('K')
            elif idx == 11:
                detected_letters.append('L')
            elif idx == 12:
                detected_letters.append('M')
            elif idx == 13:
                detected_letters.append('N')
            elif idx == 14:
                detected_letters.append('O')
            elif idx == 15:
                detected_letters.append('P')
            elif idx == 16:
                detected_letters.append('R')
            elif idx == 17:
                detected_letters.append('S')
            elif idx == 18:
                detected_letters.append('T')
            elif idx == 19:
                detected_letters.append('U')
            elif idx == 20:
                detected_letters.append('W')
            elif idx == 21:
                detected_letters.append('X')
            elif idx == 22:
                detected_letters.append('V')
            elif idx == 23:
                detected_letters.append('Y')
            elif idx == 24:
                detected_letters.append('Z')
            elif idx == 25:
                detected_letters.append('0')
            elif idx == 26:
                detected_letters.append('1')
            elif idx == 27:
                detected_letters.append('2')
            elif idx == 28:
                detected_letters.append('3')
            elif idx == 29:
                detected_letters.append('4')
            elif idx == 30:
                detected_letters.append('5')
            elif idx == 31:
                detected_letters.append('6')
            elif idx == 32:
                detected_letters.append('7')
            elif idx == 33:
                detected_letters.append('8')
            elif idx == 34:
                detected_letters.append('9')
    # print(detected_letters)
    print("\nLicense Plate Number " + str(index) + ": ", end=" ")
    for letter in detected_letters:
        print(letter, end="")

cv.waitKey(0)
cv.destroyAllWindows()
