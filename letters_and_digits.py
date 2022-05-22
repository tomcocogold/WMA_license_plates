import cv2 as cv

path = 'lettersanddigits.png'
image = cv.imread(path)
img = cv.resize(image, (1000, 374), interpolation=cv.INTER_CUBIC)
# print('This is image number {}'.format(img_number))
# cv.imshow("Photo", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray Scale', gray)

filtered = cv.bilateralFilter(gray, 5, 200, 200)
# cv.imshow('Bilateral Filter', filtered)

threshold = cv.adaptiveThreshold(filtered, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 155, +30)
# cv.imshow("Threshold", threshold)

canny = cv.Canny(threshold, 150, 200, 0, 3, True)
# cv.imshow('Canny Edges', canny)

letters_and_digits, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# print('{} contour(s) found!'.format(len(letters_and_digits)))

# img_copy = img.copy()
# cv.drawContours(img_copy, letters_and_digits, -1, (0, 255, 0), 3)
# cv.imshow('Contours', img_copy)

cv.waitKey(0)
cv.destroyAllWindows()