import sys, pickle, cv2, os, time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.metrics import accuracy_score # Accuracy metrics for the model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Model
with open(r'rf100.pkl', 'rb') as f:
    model = pickle.load(f)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)

        while self._run_flag:
            
            ret, image = cap.read()
            results = pose.process(image)
            mp_drawing.draw_landmarks(image, results.pose_landmarks)
            
            try: 
                # Extract Pose landmarks
                data = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in data]).flatten())
                del pose_row[0:40]

                # Make Detections
                X = pd.DataFrame([pose_row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                if body_language_prob[0] > 0.55:
                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    
                    # Display Class
                    cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
                    cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    pass
                    # cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    # cv2.putText(image, 'No Detection', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass
            if ret:
                self.change_pixmap_signal.emit(image)
        # shut down capture system
        cap.release()
        
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class RealTime(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 1000, 840)
        self.setWindowTitle('Human Activity Detection')
        self.setMinimumSize(QtCore.QSize(1000, 840))
        self.setMaximumSize(QtCore.QSize(1000, 840))
        self.initUI()
        
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.label_2.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def initUI(self):
        self.w = None
        self.x = None
        # About Action
        aboutAct = QAction('About', self)
        aboutAct.setShortcut('Ctrl+1')
        aboutAct.setStatusTip('About Page')
        aboutAct.triggered.connect(self.show_about)

        # Help Action
        helpAct = QAction('Help', self)
        helpAct.setShortcut('Ctrl+2')
        helpAct.setStatusTip('Help Page')
        helpAct.triggered.connect(self.show_help)

        # Quit Action
        exitAct = QAction('Exit', self)
        exitAct.setShortcut('Q')
        exitAct.setStatusTip('Exit pplication')
        exitAct.triggered.connect(qApp.quit)

        # Menu Bar
        self.statusBar()
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('Files')
        fileMenu.addAction(aboutAct)
        fileMenu.addAction(helpAct)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAct)

        # Label RealTime Detection
        self.label = QtWidgets.QLabel(self)
        self.label.setText("Real Time Detection")
        self.label.setGeometry(QtCore.QRect(0, 20, 1000, 41))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAutoFillBackground(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Camera
        self.label_2 = QtWidgets.QLabel(self)
        self.display_width = 960
        self.display_height = 720
        self.label_2.setGeometry(QtCore.QRect(20, 80, self.display_width, self.display_height))
        self.label_2.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.label_2.setLineWidth(0)
        self.label_2.setText("")
        self.label_2.resize(self.display_width, self.display_height)
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.show()
    
    def show_about(self):
        if self.w is None:
            self.w = About()
        self.w.show()

    def show_help(self):
        if self.x is None:
            self.x = Help()
        self.x.show()

class About(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setWindowTitle('About Page')
        self.setGeometry(0, 0, 640, 360)
        self.setMinimumSize(QtCore.QSize(640, 360))
        self.setMaximumSize(QtCore.QSize(640, 360))

        # Label About Page
        self.label = QtWidgets.QLabel(self)
        self.label.setText("About")
        self.label.setGeometry(QtCore.QRect(0, 10, 641, 41))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAutoFillBackground(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Label Image
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(20, 70, 151, 191))
        self.label_2.setPixmap(QtGui.QPixmap("E:\Program Skripsi\Human Activity Dectection\img\Image1.jpg"))
        self.label_2.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.label_2.setScaledContents(True)

        # Label Box
        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setGeometry(QtCore.QRect(190, 70, 431, 261))
        self.label_3.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_3.setText("")
        self.label_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)

        # Label About
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(200, 80, 411, 121))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setWordWrap(True)
        self.label_4.setText("Aplikasi pendeteksi aktivitas dirancang untuk menganalisis video masukan dan mengklasifikasikan aktivitas yang teridentifikasi. Aplikasi ini saat ini hanya mendukung satu pengguna.")

        # Label Name
        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setGeometry(QtCore.QRect(200, 230, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setText("Nama Pembuat:")

        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(200, 250, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_6.setFont(font)
        self.label_6.setText("Evander Christian Dumalang")

        self.label_7 = QtWidgets.QLabel(self)
        self.label_7.setGeometry(QtCore.QRect(200, 270, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_7.setFont(font)
        self.label_7.setText("Lina, S.T., M. Kom., Ph.D.,")

class Help(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setWindowTitle('Help Page')
        self.setGeometry(0, 0, 840, 480)
        self.setMinimumSize(QtCore.QSize(840, 480))
        self.setMaximumSize(QtCore.QSize(840, 480))

        # Label About Page
        self.label = QtWidgets.QLabel(self)
        self.label.setText("Help")
        self.label.setGeometry(QtCore.QRect(0, 10, 840, 41))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAutoFillBackground(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Label Box
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(20, 80, 391, 371))
        self.label_2.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.label_2.setText("")
        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setGeometry(QtCore.QRect(430, 80, 391, 371))
        self.label_3.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.label_3.setText("")

        # Label Text
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(30, 90, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setText("Cara Penggunaan:")

        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setGeometry(QtCore.QRect(30, 130, 371, 101))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_5.setFont(font)
        self.label_5.setWordWrap(True)
        self.label_5.setText("1. Nyalakan Camera Webcam/Hubungkan Secara Langsung (Jika error bisa dijalankan ulang file ini)")

        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(30, 230, 371, 101))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_6.setFont(font)
        self.label_6.setWordWrap(True)
        self.label_6.setText("2. Aktvitas Kelas dan Probabilitas dari Aktivitas yang akan diklasifikasikan adalah (Berdiri, Berjalan dan Mengambil)")

        self.label_7 = QtWidgets.QLabel(self)
        self.label_7.setGeometry(QtCore.QRect(440, 90, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_7.setFont(font)
        self.label_7.setText("Jika ada Masalah:")

        self.label_8 = QtWidgets.QLabel(self)
        self.label_8.setGeometry(QtCore.QRect(440, 110, 371, 101))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_8.setFont(font)
        self.label_8.setWordWrap(True)
        self.label_8.setText("1. Hubungi Email: Evander.535180125@stu.untar.ac.d")

def main():
    app = QApplication(sys.argv)
    ex = RealTime()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()