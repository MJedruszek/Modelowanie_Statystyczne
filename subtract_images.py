import cv2

img1 = cv2.imread('images/cat.png')
img2 = cv2.imread('images/cat9.png')

#odejmowanie zwykłe
subtraction = cv2.subtract(img1, img2)
boosted_sub = cv2.normalize(subtraction, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

#różnica jako wartość bezwzględna
diff = cv2.absdiff(img1, img2)
boosted_diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

cv2.imshow('Subtracted (img1 - img2)', boosted_sub)
cv2.imshow('Absolute Difference', boosted_diff)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('images/subcat19.png', boosted_sub)
cv2.imwrite('images/abscat19.png', boosted_diff)