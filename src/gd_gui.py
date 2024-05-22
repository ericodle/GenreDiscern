import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QLineEdit, QInputDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from subprocess import Popen, PIPE

import MFCC_extraction, models, model_sort, train_model

class WelcomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome")
        self.setGeometry(100, 100, 600, 400)

        # Add label for the title "GenreDiscern"
        self.label_title = QLabel("<html><b>GenreDiscern</b></html>", self)
        self.label_title.setGeometry(200, 50, 200, 30)

        # Add label for authors and version
        self.label_info = QLabel("By: Eric and Rebecca <br>Version: 1.0.0", self)
        self.label_info.setGeometry(200, 80, 200, 50)

        # Add image label
        self.label_image = QLabel(self)
        self.label_image.setGeometry(150, 130, 300, 200)  # Adjust position and size as needed
        pixmap = QPixmap('img/gd_logo.png').scaled(300, 300)  # Scale the image to fit the label
        self.label_image.setPixmap(pixmap)

        # Add Start button
        self.button_start = QPushButton("Start", self)
        self.button_start.setGeometry(200, 340, 100, 30)  # Adjust position as needed
        self.button_start.clicked.connect(self.start_application)

        # Add Quit button
        self.button_quit = QPushButton("Quit", self)
        self.button_quit.setGeometry(320, 340, 100, 30)  # Adjust position as needed
        self.button_quit.clicked.connect(QApplication.instance().quit)

    def start_application(self):
        self.close()  # Close the welcome window
        self.hub_window = HubWindow()
        self.hub_window.show()

class HubWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hub")
        self.setGeometry(100, 100, 600, 400)

        self.label_hub = QLabel("Choose an option:", self)
        self.label_hub.setGeometry(200, 50, 200, 30)

        self.button_mfcc_extraction = QPushButton("MFCC Extraction", self)
        self.button_mfcc_extraction.setGeometry(200, 100, 200, 30)
        self.button_mfcc_extraction.clicked.connect(self.open_mfcc_extraction)

        self.button_train_model = QPushButton("Train Model", self)
        self.button_train_model.setGeometry(200, 150, 200, 30)
        self.button_train_model.clicked.connect(self.open_train_model)

        self.button_execute_sort = QPushButton("Execute Sort", self)
        self.button_execute_sort.setGeometry(200, 200, 200, 30)
        self.button_execute_sort.clicked.connect(self.open_execute_sort)

    def open_mfcc_extraction(self):
        self.close()  # Close the hub window
        self.preprocess_mfcc_window = PreprocessMFCCWindow()
        self.preprocess_mfcc_window.show()

    def open_train_model(self):
        self.close()  # Close the hub window
        self.train_model_window = TrainModelWindow()
        self.train_model_window.show()

    def open_execute_sort(self):
        self.close()  # Close the hub window
        self.execute_sort_window = ExecuteSortWindow()
        self.execute_sort_window.show()

class PreprocessMFCCWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Preprocess MFCCs")
        self.setGeometry(100, 100, 600, 400)

        self.dataset_path = ""
        self.output_path = ""
        self.filename = ""

        # Labels
        self.label_dataset = QLabel("Select Music Dataset Directory:", self)
        self.label_dataset.setGeometry(50, 50, 250, 30)

        self.label_output = QLabel("Select Output Directory:", self)
        self.label_output.setGeometry(50, 100, 250, 30)

        self.label_filename = QLabel("Enter Filename:", self)
        self.label_filename.setGeometry(50, 150, 250, 30)

        self.label_message = QLabel("", self)
        self.label_message.setGeometry(200, 250, 250, 30)
        self.label_message.setAlignment(Qt.AlignCenter)

        # Line Edits
        self.line_edit_dataset = QLineEdit(self)
        self.line_edit_dataset.setGeometry(300, 50, 250, 30)
        self.line_edit_dataset.setReadOnly(True)

        self.line_edit_output = QLineEdit(self)
        self.line_edit_output.setGeometry(300, 100, 250, 30)
        self.line_edit_output.setReadOnly(True)

        self.line_edit_filename = QLineEdit(self)
        self.line_edit_filename.setGeometry(300, 150, 250, 30)

        # Buttons
        self.button_browse_dataset = QPushButton("Go", self)
        self.button_browse_dataset.setGeometry(560, 50, 30, 30)
        self.button_browse_dataset.clicked.connect(self.browse_dataset)

        self.button_browse_output = QPushButton("Go", self)
        self.button_browse_output.setGeometry(560, 100, 30, 30)
        self.button_browse_output.clicked.connect(self.browse_output)

        self.button_start_preprocessing = QPushButton("Start Pre-processing", self)
        self.button_start_preprocessing.setGeometry(200, 200, 200, 40)
        self.button_start_preprocessing.clicked.connect(self.start_preprocessing)

        self.back_button = QPushButton("Back", self)
        self.back_button.setGeometry(250, 320, 100, 30)
        self.back_button.clicked.connect(self.back_to_hub)

    def browse_dataset(self):
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Select Music Dataset Directory")
        self.line_edit_dataset.setText(self.dataset_path)

    def browse_output(self):
        self.output_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        self.line_edit_output.setText(self.output_path)

    def start_preprocessing(self):
        self.filename = self.line_edit_filename.text()
        if not self.dataset_path or not self.output_path or not self.filename:
            print("Please fill in all fields.")
            return

        # Call MFCC_extraction.py script with the provided arguments
        process = Popen(["python3", "src/MFCC_extraction.py", self.dataset_path, self.output_path, self.filename], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        print(stdout.decode())
        print(stderr.decode())

        self.label_message.setText("Pre-processing finished!")

    def back_to_hub(self):
        self.close()  # Close the preprocess MFCC window
        self.hub_window = HubWindow()
        self.hub_window.show()

class TrainModelWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Train Model")
        self.setGeometry(100, 100, 600, 400)

        self.label_train_model = QLabel("Train Model", self)
        self.label_train_model.setGeometry(200, 50, 200, 30)

        self.button_select_mfcc_path = QPushButton("Select MFCC path", self)
        self.button_select_mfcc_path.setGeometry(200, 50, 200, 30)
        self.button_select_mfcc_path.clicked.connect(self.select_mfcc_path)

        self.button_select_model_type = QPushButton("Select Model Type", self)
        self.button_select_model_type.setGeometry(200, 100, 200, 30)
        self.button_select_model_type.clicked.connect(self.select_model_type)

        self.button_training_output_path = QPushButton("Select Output Directory", self)
        self.button_training_output_path.setGeometry(200, 150, 200, 30)
        self.button_training_output_path.clicked.connect(self.select_output_directory)

        self.button_initial_lr = QPushButton("Enter Initial Learning Rate", self)
        self.button_initial_lr.setGeometry(200, 200, 200, 30)
        self.button_initial_lr.clicked.connect(self.enter_initial_lr)

        self.button_train_model = QPushButton("Train Model!", self)
        self.button_train_model.setGeometry(200, 250, 200, 30)
        self.button_train_model.clicked.connect(self.train_model)

        self.back_button = QPushButton("Back", self)
        self.back_button.setGeometry(250, 300, 100, 30)
        self.back_button.clicked.connect(self.back_to_hub)

        # Add label_message
        self.label_message = QLabel("", self)
        self.label_message.setGeometry(200, 350, 200, 30)
        self.label_message.setAlignment(Qt.AlignCenter)

        # Attributes to store user inputs
        self.mfcc_path = ""
        self.model_type = ""
        self.output_directory = ""
        self.initial_lr_value = 0.0001

    def select_mfcc_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSON File", "", "JSON Files (*.json)")
        if file_path:
            self.mfcc_path = file_path
            print("Selected JSON file:", self.mfcc_path)

    def select_model_type(self):
        model_type, ok = QInputDialog.getItem(self, "Select Model Type", "Model Types", ["FC", "CNN", "LSTM", "GRU", "Tr_FC", "Tr_CNN", "Tr_LSTM", "Tr_GRU"], 0, False)
        if ok:
            self.model_type = model_type
            print("Selected model type:", self.model_type)

    def select_output_directory(self):
        self.output_directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if self.output_directory:
            print("Selected output directory:", self.output_directory)

    def enter_initial_lr(self):
        initial_lr, ok = QInputDialog.getDouble(self, "Enter Initial Learning Rate", "Initial Learning Rate:", 0.0001, 0, 10, 6)
        if ok:
            self.initial_lr_value = initial_lr
            print("Entered initial learning rate:", self.initial_lr_value)

    def train_model(self):
        if not self.mfcc_path or not self.model_type or not self.output_directory:
            print("Please fill in all fields.")
            return

        # Call train_model.py script with the provided arguments
        process = Popen(["python3", "src/train_model.py", self.mfcc_path, self.model_type, self.output_directory, str(self.initial_lr_value)], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        print(stdout.decode())
        print(stderr.decode())

        # Show message box indicating training finished
        QMessageBox.information(self, "Training Finished", "Training finished!")

    def back_to_hub(self):
        self.close()  # Close the train model window
        self.hub_window = HubWindow()
        self.hub_window.show()

class ExecuteSortWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Execute Sort")
        self.setGeometry(100, 100, 600, 400)

        self.label_execute_sort = QLabel("Execute Sort using Pretrained Model", self)
        self.label_execute_sort.setGeometry(150, 50, 300, 30)

        self.back_button = QPushButton("Back", self)
        self.back_button.setGeometry(250, 200, 100, 30)
        self.back_button.clicked.connect(self.back_to_hub)

    def back_to_hub(self):
        self.close()  # Close the execute sort window
        self.hub_window = HubWindow()
        self.hub_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    welcome_window = WelcomeWindow()
    welcome_window.show()
    sys.exit(app.exec_())
