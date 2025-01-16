import cv2
import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
import sqlite3
import pyttsx3
from threading import Thread
from tkinter import messagebox
from threading import Thread
from PIL import Image, ImageTk


# Load your Keras model
model = load_model('cnn_model_keras2.h5')

engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def get_image_size():
    img = cv2.imread('gestures/0/100.jpg', 0)
    return img.shape

image_x, image_y = get_image_size()

def get_pred_from_contour(contour, thresh):
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = thresh[y1:y1+h1, x1:x1+w1]
    text = ""
    if w1 > h1:
        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    elif h1 > w1:
        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0, 0, 0))
    pred_probab, pred_class = keras_predict(model, save_img)
    if pred_probab*100 > 70:
        text = get_pred_text_from_db(pred_class)
    return text


def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
    cursor = conn.execute(cmd)
    for row in cursor:
        return row[0]

def get_operator(pred_text):
    try:
        pred_text = int(pred_text)
    except:
        return ""
    operator = ""
    if pred_text == 1:
        operator = "+"
    elif pred_text == 2:
        operator = "-"
    elif pred_text == 3:
        operator = "*"
    elif pred_text == 4:
        operator = "/"
    elif pred_text == 5:
        operator = "%"
    elif pred_text == 6:
        operator = "**"
    elif pred_text == 7:
        operator = ">>"
    elif pred_text == 8:
        operator = "<<"
    elif pred_text == 9:
        operator = "&"
    elif pred_text == 0:
        operator = "|"
    return operator

hist = get_hand_hist()
x, y, w, h = 300, 100, 300, 300
is_voice_on = True

def get_img_contour_thresh(img):
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cv2.filter2D(dst,-1,disc,dst)
    blur = cv2.GaussianBlur(dst, (11,11), 0)
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = cv2.merge((thresh,thresh,thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thresh = thresh[y:y+h, x:x+w]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    return img, contours, thresh

def say_text(text):
    if not is_voice_on:
        return
    while engine._inLoop:
        pass
    engine.say(text)
    engine.runAndWait()

def calculator_mode(cam):
    global is_voice_on
    flag = {"first": False, "operator": False, "second": False, "clear": False}
    count_same_frames = 0
    first, operator, second = "", "", ""
    pred_text = ""
    calc_text = ""
    info = "Enter first number"
    Thread(target=say_text, args=(info,)).start()
    count_clear_frames = 0
    while True:
        img = cam.read()[1]
        img = cv2.resize(img, (640, 480))
        img, contours, thresh = get_img_contour_thresh(img)
        old_pred_text = pred_text
        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                pred_text = get_pred_from_contour(contour, thresh)
                if old_pred_text == pred_text:
                    count_same_frames += 1
                else:
                    count_same_frames = 0

                if pred_text == "C":
                    if count_same_frames > 5:
                        count_same_frames = 0
                        first, second, operator, pred_text, calc_text = '', '', '', '', ''
                        flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
                        info = "Enter first number"
                        Thread(target=say_text, args=(info,)).start()

                elif pred_text == "Best of Luck " and count_same_frames > 15:
                    count_same_frames = 0
                    if flag['clear']:
                        first, second, operator, pred_text, calc_text = '', '', '', '', ''
                        flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
                        info = "Enter first number"
                        Thread(target=say_text, args=(info,)).start()
                    elif second != '':
                        flag['second'] = True
                        info = "Clear screen"
                        second = ''
                        flag['clear'] = True
                        try:
                            calc_text += "= "+str(eval(calc_text))
                        except:
                            calc_text = "Invalid operation"
                        if is_voice_on:
                            speech = calc_text
                            speech = speech.replace('-', ' minus ')
                            speech = speech.replace('/', ' divided by ')
                            speech = speech.replace('**', ' raised to the power ')
                            speech = speech.replace('*', ' multiplied by ')
                            speech = speech.replace('%', ' mod ')
                            speech = speech.replace('>>', ' bitwise right shift ')
                            speech = speech.replace('<<', ' bitwise left shift ')
                            speech = speech.replace('&', ' bitwise and ')
                            speech = speech.replace('|', ' bitwise or ')
                            Thread(target=say_text, args=(speech,)).start()
                    elif first != '':
                        flag['first'] = True
                        info = "Enter operator"
                        Thread(target=say_text, args=(info,)).start()
                        first = ''

                elif pred_text != "Best of Luck " and pred_text.isnumeric():
                    if flag['first'] == False:
                        if count_same_frames > 15:
                            count_same_frames = 0
                            Thread(target=say_text, args=(pred_text,)).start()
                            first += pred_text
                            calc_text += pred_text
                    elif flag['operator'] == False:
                        operator = get_operator(pred_text)
                        if count_same_frames > 15:
                            count_same_frames = 0
                            flag['operator'] = True
                            calc_text += operator
                            info = "Enter second number"
                            Thread(target=say_text, args=(info,)).start()
                            operator = ''
                    elif flag['second'] == False:
                        if count_same_frames > 15:
                            Thread(target=say_text, args=(pred_text,)).start()
                            second += pred_text
                            calc_text += pred_text
                            count_same_frames = 0

        if count_clear_frames == 30:
            first, second, operator, pred_text, calc_text = '', '', '', '', ''
            flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
            info = "Enter first number"
            Thread(target=say_text, args=(info,)).start()
            count_clear_frames = 0

        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Calculator Mode", (100, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
        cv2.putText(blackboard, "Predicted text- " + pred_text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, "Operator " + operator, (30, 140), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
        cv2.putText(blackboard, calc_text, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        cv2.putText(blackboard, info, (30, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255) )
        if is_voice_on:
            cv2.putText(blackboard, "Voice on", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
        else:
            cv2.putText(blackboard, "Voice off", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        cv2.imshow("thresh", thresh)
        keypress = cv2.waitKey(1)
        if keypress == ord('q') or keypress == ord('t'):
            break
        if keypress == ord('v') and is_voice_on:
            is_voice_on = False
        elif keypress == ord('v') and not is_voice_on:
            is_voice_on = True

    if keypress == ord('t'):
        return 1
    else:
        return 0

def text_mode(cam):
    global is_voice_on
    text = ""
    word = ""
    count_same_frame = 0
    while True:
        img = cam.read()[1]
        img = cv2.resize(img, (640, 480))
        img, contours, thresh = get_img_contour_thresh(img)
        old_text = text
        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                text = get_pred_from_contour(contour, thresh)
                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0

                if count_same_frame > 20:
                    if len(text) == 1:
                        Thread(target=say_text, args=(text, )).start()
                    word = word + text
                    if word.startswith('I/Me '):
                        word = word.replace('I/Me ', 'I ')
                    elif word.endswith('I/Me '):
                        word = word.replace('I/Me ', 'me ')
                    count_same_frame = 0

            elif cv2.contourArea(contour) < 1000:
                if word != '':
                    Thread(target=say_text, args=(word, )).start()
                text = ""
                word = ""
        else:
            if word != '':
                Thread(target=say_text, args=(word, )).start()
            text = ""
            word = ""
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Text Mode", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
        cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        if is_voice_on:
            cv2.putText(blackboard, "Voice on", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
        else:
            cv2.putText(blackboard, "Voice off", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        cv2.imshow("thresh", thresh)
        keypress = cv2.waitKey(1)
        if keypress == ord('q') or keypress == ord('c'):
            break
        if keypress == ord('v') and is_voice_on:
            is_voice_on = False
        elif keypress == ord('v') and not is_voice_on:
            is_voice_on = True

    if keypress == ord('c'):
        return 2
    else:
        return 0


class GestureRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition GUI")
        self.img_path = "signlang.png"
        self.create_gui()

    def create_gui(self):
        # Load and display the image
        img = Image.open(self.img_path)
        img = img.resize((400, 400), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        label_img = tk.Label(self.root, image=img)
        label_img.image = img
        label_img.pack(pady=10)

        with_hist_button = tk.Button(self.root, text="With Histogram", command=self.with_histogram, padx=10, pady=5)
        with_hist_button.pack(pady=10)

        without_hist_button = tk.Button(self.root, text="Without Histogram", command=self.without_histogram, padx=10, pady=5)
        without_hist_button.pack(pady=10)

        back_button = tk.Button(self.root, text="Back", command=self.stop_recognition, padx=10, pady=5)
        back_button.pack(pady=10)

    def with_histogram(self):
        self.recognition_thread = Thread(target=self.start_recognition, args=(True,))
        self.recognition_thread.start()

    def without_histogram(self):
        self.recognition_thread = Thread(target=self.start_recognition, args=(False,))
        self.recognition_thread.start()

    def stop_recognition(self):
        cv2.destroyAllWindows()
        if hasattr(self, 'recognition_thread') and self.recognition_thread.is_alive():
            self.recognition_thread.join()

    def start_recognition(self, use_histogram):
            def recognize():
                cam = cv2.VideoCapture(0)
                if cam.read()[0] == False:
                    cam = cv2.VideoCapture(0)
                text = ""
                word = ""
                count_same_frame = 0
                keypress = 1
                while True:
                    if keypress == 1:
                        keypress = text_mode(cam)
                    elif keypress == 2:
                        keypress = calculator_mode(cam)
                    else:
                        break
            # Place the entire content of the 'recognize' function here
            cam = cv2.VideoCapture(0)
            if cam.read()[0] == False:
                cam = cv2.VideoCapture(0)
            text = ""
            word = ""
            count_same_frame = 0
            keypress = 1
            while True:
                if keypress == 1:
                    keypress = text_mode(cam)
                elif keypress == 2:
                    keypress = calculator_mode(cam)
                else:
                    break
            keras_predict(model, np.zeros((50, 50), dtype = np.uint8))        
            recognize()
      
    def text_mode(self, cam):
            global is_voice_on
            text = ""
            word = ""
            count_same_frame = 0
            while True:
                img = cam.read()[1]
                img = cv2.resize(img, (640, 480))
                img, contours, thresh = get_img_contour_thresh(img)
                old_text = text
                if len(contours) > 0:
                    contour = max(contours, key = cv2.contourArea)
                    if cv2.contourArea(contour) > 10000:
                        text = get_pred_from_contour(contour, thresh)
                        if old_text == text:
                            count_same_frame += 1
                        else:
                            count_same_frame = 0

                        if count_same_frame > 20:
                            if len(text) == 1:
                                Thread(target=say_text, args=(text, )).start()
                            word = word + text
                            if word.startswith('I/Me '):
                                word = word.replace('I/Me ', 'I ')
                            elif word.endswith('I/Me '):
                                word = word.replace('I/Me ', 'me ')
                            count_same_frame = 0

                    elif cv2.contourArea(contour) < 1000:
                        if word != '':
                            Thread(target=say_text, args=(word, )).start()
                        text = ""
                        word = ""
                else:
                    if word != '':
                        Thread(target=say_text, args=(word, )).start()
                    text = ""
                    word = ""
                blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blackboard, "Text Mode", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
                cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
                cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
                if is_voice_on:
                    cv2.putText(blackboard, "Voice on", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
                else:
                    cv2.putText(blackboard, "Voice off", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                res = np.hstack((img, blackboard))
                cv2.imshow("Recognizing gesture", res)
                cv2.imshow("thresh", thresh)
                keypress = cv2.waitKey(1)
                if keypress == ord('q') or keypress == ord('c'):
                    break
                if keypress == ord('v') and is_voice_on:
                    is_voice_on = False
                elif keypress == ord('v') and not is_voice_on:
                    is_voice_on = True

            if keypress == ord('c'):
                return 2
            else:
                return 0
    # Implement the 'text_mode' function
            pass

    def calculator_mode(self, cam):
        # Implement the 'calculator_mode' function
        pass

    def __del__(self):
        self.stop_recognition()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureRecognitionGUI(root)
    root.mainloop()