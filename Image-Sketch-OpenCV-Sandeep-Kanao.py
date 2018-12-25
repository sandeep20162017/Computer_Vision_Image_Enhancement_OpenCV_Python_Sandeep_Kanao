import cv2

# Read image
im = cv2.imread("canada.jpg");

# Edge preserving filter with two different flags.
imout = cv2.edgePreservingFilter(im, flags=cv2.RECURS_FILTER);
cv2.imwrite("farmer-1_edge-preserving-recursive-filter.jpg", imout);

imout = cv2.edgePreservingFilter(im, flags=cv2.NORMCONV_FILTER);
cv2.imwrite("mankar_edge-preserving-normalized-convolution-filter.jpg", imout);

# Detail enhance filter
imout = cv2.detailEnhance(im);
cv2.imwrite("mankar_detail-enhance.jpg", imout);

# Pencil sketch filter
imout_gray, imout = cv2.pencilSketch(im, sigma_s=100, sigma_r=0.12, shade_factor=0.09);
cv2.imwrite("canada_pencil-sketch.jpg", imout_gray);
cv2.imwrite("canada_pencil-sketch-color.jpg", imout);

# Stylization filter
cv2.stylization(im,imout);
cv2.imwrite("canada_stylization.jpg", imout);