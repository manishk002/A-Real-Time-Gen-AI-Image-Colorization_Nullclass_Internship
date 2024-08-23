import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import torch
from interactive_colorization_model import UserGuidedColorizationNet, lab_to_rgb

class ColorizationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_model()

    def initUI(self):
        self.setWindowTitle('Interactive Image Colorization')
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()
        
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()
        self.load_button = QPushButton('Load Image')
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)

        self.colorize_button = QPushButton('Colorize')
        self.colorize_button.clicked.connect(self.colorize_image)
        button_layout.addWidget(self.colorize_button)

        layout.addLayout(button_layout)
        main_widget.setLayout(layout)

        self.image = None
        self.user_hints = None

    def load_model(self):
        self.model = UserGuidedColorizationNet()
        self.model.load_state_dict(torch.load('interactive_colorization_model.pth'))
        self.model.eval()

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', r"<Default dir>", "Image files (*.jpg *.jpeg *.png)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.display_image(self.image)
            self.user_hints = np.zeros_like(self.image)

    def display_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if self.image is not None:
            x = event.x() - self.image_label.x()
            y = event.y() - self.image_label.y()
            if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                color = QColorDialog.getColor()
                if color.isValid():
                    self.user_hints[y, x] = [color.red(), color.green(), color.blue()]
                    self.display_image(self.image * 0.7 + self.user_hints * 0.3)

    def colorize_image(self):
        if self.image is not None:
            L = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)[:,:,0]
            L = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float() / 50.0 - 1.0
            user_hints = torch.from_numpy(self.user_hints).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            with torch.no_grad():
                output = self.model(L, user_hints)

            output = output.squeeze().cpu().numpy()
            output = (output * 110.0).transpose(1, 2, 0)
            
            result = np.concatenate([L.squeeze().cpu().numpy() * 50.0 + 50.0, output], axis=2)
            result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2RGB)
            
            self.display_image(result)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ColorizationGUI()
    ex.show()
    sys.exit(app.exec_())
