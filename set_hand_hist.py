import cv2
import numpy as np
import pickle

def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    img_crop = None
    crop = None
    for i in range(10):
        for j in range(5):
            if img_crop is None:
                img_crop = img[y:y+h, x:x+w]
            else:
                img_crop = np.hstack((img_crop, img[y:y+h, x:x+w]))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            x += w + d
        if crop is None:
            crop = img_crop
        else:
            crop = np.vstack((crop, img_crop))
        img_crop = None
        x = 420
        y += h + d
    return crop

def get_hand_hist():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)

    x, y, w, h = 300, 100, 300, 300
    flag_pressed_c, flag_pressed_s = False, False
    img_crop = None
    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            hsv_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
            flag_pressed_c = True
            hist = cv2.calcHist([hsv_crop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        elif keypress == ord('s'):
            flag_pressed_s = True
            break
        if flag_pressed_c:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresh", thresh)
        if not flag_pressed_s:
            img_crop = build_squares(img)
        cv2.imshow("Set hand histogram", img)

    cam.release()
    cv2.destroyAllWindows()

    with open("hist", "wb") as f:
        pickle.dump(hist, f)

get_hand_hist()
