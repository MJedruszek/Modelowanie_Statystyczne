import cv2

img1 = cv2.imread('images/cat.png')
img2 = cv2.imread('images/cat5.png')

#odejmowanie zwykłe
subtraction = cv2.subtract(img1, img2)
boosted_sub = cv2.multiply(subtraction, 5)

#różnica jako wartość bezwzględna
diff = cv2.absdiff(img1, img2)
boosted_diff = cv2.multiply(diff, 5)

cv2.imshow('Subtracted (img1 - img2)', boosted_sub)
cv2.imshow('Absolute Difference', boosted_diff)

cv2.waitKey(0)
cv2.destroyAllWindows()