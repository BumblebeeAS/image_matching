import cv2


@staticmethod
def equalize_green_blue(img):
    b, g, r = cv2.split(img)

    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)

    img_eq = cv2.merge((b_eq, g_eq, r))

    return img_eq
