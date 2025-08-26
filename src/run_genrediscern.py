########################################################################
# IMPORT LIBRARIES
########################################################################

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, 
    QLineEdit, QInputDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QProgressBar, QTextEdit, QGroupBox, QGridLayout, QSpacerItem,
    QSizePolicy, QFrame, QScrollArea, QMessageBox, QComboBox,
    QSlider, QCheckBox, QTabWidget, QSplitter
)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QPainter
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
import MFCC_extraction, train_model

########################################################################
# TRAINING WORKER CLASS
########################################################################

class TrainingWorker(QThread):
    """Worker thread for training to prevent GUI freezing"""
    training_started = pyqtSignal()
    training_finished = pyqtSignal(object)  # Pass result or exception
    training_error = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, mfcc_path, model_type, output_directory, initial_lr, batch_size):
        super().__init__()
        self.mfcc_path = mfcc_path
        self.model_type = model_type
        self.output_directory = output_directory
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        
    def run(self):
        """Run training in background thread"""
        try:
            self.training_started.emit()
            self.progress_update.emit("Starting training...")
            
            # Call the actual training function
            result = train_model.main(
                self.mfcc_path, 
                self.model_type, 
                self.output_directory, 
                self.initial_lr, 
                self.batch_size
            )
            
            self.progress_update.emit("Training completed successfully!")
            self.training_finished.emit(result)
            
        except Exception as e:
            error_msg = str(e)
            self.progress_update.emit(f"Training error: {error_msg}")
            self.training_error.emit(error_msg)

########################################################################
# WINDOW SYSTEM
########################################################################

# Get the directory path of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))
containing_dir = os.path.dirname(base_dir)

class StyledButton(QPushButton):
    """Custom styled button with hover effects"""
    def __init__(self, text, primary_color="#4CAF50", hover_color="#45a049"):
        super().__init__(text)
        self.primary_color = primary_color
        self.hover_color = hover_color
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {primary_color};
                border: none;
                color: white;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: #3d8b40;
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
        """)

class InfoLabel(QLabel):
    """Informative label with styling"""
    def __init__(self, text, info_text="", is_title=False):
        super().__init__(text)
        if is_title:
            self.setStyleSheet("""
                QLabel {
                    color: #2c3e50;
                    font-size: 24px;
                    font-weight: bold;
                    padding: 10px;
                    background-color: #ecf0f1;
                    border-radius: 8px;
                    border: 2px solid #bdc3c7;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    color: #34495e;
                    font-size: 12px;
                    padding: 5px;
                    background-color: #f8f9fa;
                    border-radius: 4px;
                    border: 1px solid #dee2e6;
                }
            """)
        
        if info_text:
            self.setToolTip(info_text)

class StatusBar(QFrame):
    """Custom status bar with progress and information"""
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 3px;
                text-align: center;
                background-color: #e9ecef;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 2px;
            }
        """)
        
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.progress_bar)

class WelcomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenreDiscern - Music Genre Classification System")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
            }
        """)
        
        # Central widget with layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Title section
        title_frame = QFrame()
        title_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 20px;
            }
        """)
        title_layout = QVBoxLayout(title_frame)
        
        # Logo and title
        logo_layout = QHBoxLayout()
        
        # Logo
        logo_label = QLabel()
        image_path = os.path.join(containing_dir, 'img', 'gd_logo.png')
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path).scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        logo_layout.addWidget(logo_label)
        
        # Title text
        title_text = QLabel("GenreDiscern")
        title_text.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 48px;
                font-weight: bold;
                font-family: 'Arial', sans-serif;
            }
        """)
        logo_layout.addWidget(title_text)
        logo_layout.addStretch()
        
        title_layout.addLayout(logo_layout)
        
        # Subtitle
        subtitle = QLabel("Advanced Music Genre Classification via Deep Learning")
        subtitle.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                font-size: 18px;
                font-style: italic;
            }
        """)
        subtitle.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(subtitle)
        
        main_layout.addWidget(title_frame)
        
        # Info section
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                padding: 20px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        
        # Features
        features_label = QLabel("🎵 Features:")
        features_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        info_layout.addWidget(features_label)
        
        features_text = QLabel("""
        • MFCC Feature Extraction from Audio Files
        • Multiple Neural Network Architectures (CNN, LSTM, xLSTM, GRU, Transformers)
        • Advanced Training with Early Stopping and Learning Rate Scheduling
        • Comprehensive Model Evaluation and Visualization
        • User-Friendly Interface for Music Genre Classification
        """)
        features_text.setStyleSheet("color: #34495e; font-size: 14px; line-height: 1.5;")
        info_layout.addWidget(features_text)
        
        # Authors and version
        authors_label = QLabel("👨‍💻 Developed by Eric Odle and Professor Rebecca Lin at Feng Chia University | 🚀 Version: 0.1.0")
        authors_label.setStyleSheet("color: #7f8c8d; font-size: 14px; font-weight: bold;")
        authors_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(authors_label)
        
        main_layout.addWidget(info_frame)
        
        # Buttons section
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(20)
        
        self.button_start = StyledButton("🚀 Start Application", "#3498db", "#2980b9")
        self.button_start.setMinimumHeight(50)
        self.button_start.clicked.connect(self.start_application)
        
        self.button_quit = StyledButton("❌ Quit", "#e74c3c", "#c0392b")
        self.button_quit.setMinimumHeight(50)
        self.button_quit.clicked.connect(QApplication.instance().quit)
        
        buttons_layout.addWidget(self.button_start)
        buttons_layout.addWidget(self.button_quit)
        
        main_layout.addLayout(buttons_layout)
        main_layout.addStretch()

    def start_application(self):
        self.close()
        self.hub_window = HubWindow()
        self.hub_window.show()

class HubWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenreDiscern - Main Hub")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_label = QLabel("🎯 Choose Your Operation")
        header_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 28px;
                font-weight: bold;
                padding: 20px;
                background-color: #ecf0f1;
                border-radius: 10px;
                border: 2px solid #bdc3c7;
            }
        """)
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)
        
        # Options grid
        options_layout = QGridLayout()
        options_layout.setSpacing(20)
        
        # MFCC Extraction
        mfcc_card = self.create_option_card(
            "🎵 MFCC Extraction",
            "Extract Mel-Frequency Cepstral Coefficients from your music dataset",
            "Process audio files to extract features for training",
            "#3498db"
        )
        mfcc_card.clicked.connect(self.open_mfcc_extraction)
        options_layout.addWidget(mfcc_card, 0, 0)
        
        # Train Model
        train_card = self.create_option_card(
            "🧠 Train Model",
            "Train neural networks on extracted MFCC features",
            "Choose from multiple architectures and hyperparameters",
            "#e74c3c"
        )
        train_card.clicked.connect(self.open_train_model)
        options_layout.addWidget(train_card, 0, 1)
        
        main_layout.addLayout(options_layout)
        
        # Status bar
        self.status_bar = StatusBar()
        main_layout.addWidget(self.status_bar)
        
        # Back button
        back_button = StyledButton("← Back to Welcome", "#95a5a6", "#7f8c8d")
        back_button.clicked.connect(self.back_to_welcome)
        main_layout.addWidget(back_button)

    def create_option_card(self, title, description, details, color):
        """Create a clickable option card"""
        card = QPushButton()  # Changed from QFrame to QPushButton
        card.setStyleSheet(f"""
            QPushButton {{
                background-color: white;
                border: 2px solid {color};
                border-radius: 10px;
                padding: 20px;
                min-height: 150px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: #f8f9fa;
                border-width: 3px;
            }}
            QPushButton:pressed {{
                background-color: #e9ecef;
            }}
        """)
        card.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(card)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        layout.addWidget(title_label)
        
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #2c3e50; font-size: 14px; margin: 10px 0;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        details_label = QLabel(details)
        details_label.setStyleSheet("color: #7f8c8d; font-size: 12px; font-style: italic;")
        details_label.setWordWrap(True)
        layout.addWidget(details_label)
        
        layout.addStretch()
        
        return card

    def open_mfcc_extraction(self):
        self.close()
        self.preprocess_mfcc_window = PreprocessMFCCWindow()
        self.preprocess_mfcc_window.show()

    def open_train_model(self):
        self.close()
        self.train_model_window = TrainModelWindow()
        self.train_model_window.show()

    def back_to_welcome(self):
        self.close()
        self.welcome_window = WelcomeWindow()
        self.welcome_window.show()

class PreprocessMFCCWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenreDiscern - MFCC Feature Extraction")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
        """)
        
        self.dataset_path = ""
        self.output_path = ""
        self.filename = ""
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_label = QLabel("🎵 MFCC Feature Extraction")
        header_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                padding: 15px;
                background-color: #ecf0f1;
                border-radius: 8px;
                border: 2px solid #bdc3c7;
            }
        """)
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel("Extract Mel-Frequency Cepstral Coefficients from your music dataset for training neural networks.")
        desc_label.setStyleSheet("color: #34495e; font-size: 14px; padding: 10px;")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        main_layout.addWidget(desc_label)
        
        # Input form
        form_group = QGroupBox("📁 Input Parameters")
        form_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        form_layout = QGridLayout(form_group)
        form_layout.setSpacing(15)
        
        # Dataset path
        dataset_label = QLabel("🎼 Music Dataset Directory:")
        dataset_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        form_layout.addWidget(dataset_label, 0, 0)
        
        self.line_edit_dataset = QLineEdit()
        self.line_edit_dataset.setPlaceholderText("Select directory containing music files...")
        self.line_edit_dataset.setReadOnly(True)
        self.line_edit_dataset.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                font-size: 14px;
            }
        """)
        form_layout.addWidget(self.line_edit_dataset, 0, 1)
        
        self.button_browse_dataset = StyledButton("Browse", "#3498db", "#2980b9")
        self.button_browse_dataset.clicked.connect(self.browse_dataset)
        form_layout.addWidget(self.button_browse_dataset, 0, 2)
        
        # Output path
        output_label = QLabel("📤 Output Directory:")
        output_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        form_layout.addWidget(output_label, 1, 0)
        
        self.line_edit_output = QLineEdit()
        self.line_edit_output.setPlaceholderText("Select directory to save extracted features...")
        self.line_edit_output.setReadOnly(True)
        self.line_edit_output.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                font-size: 14px;
            }
        """)
        form_layout.addWidget(self.line_edit_output, 1, 1)
        
        self.button_browse_output = StyledButton("Browse", "#3498db", "#2980b9")
        self.button_browse_output.clicked.connect(self.browse_output)
        form_layout.addWidget(self.button_browse_output, 1, 2)
        
        # Filename
        filename_label = QLabel("📝 Output Filename:")
        filename_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        form_layout.addWidget(filename_label, 2, 0)
        
        self.line_edit_filename = QLineEdit()
        self.line_edit_filename.setPlaceholderText("Enter filename for extracted features (e.g., features.json)")
        self.line_edit_filename.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                font-size: 14px;
            }
        """)
        form_layout.addWidget(self.line_edit_filename, 2, 1)
        
        main_layout.addWidget(form_group)
        
        # Progress section
        progress_group = QGroupBox("⚡ Processing Status")
        progress_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                background-color: #ecf0f1;
                min-height: 25px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to process")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        main_layout.addWidget(progress_group)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(20)
        
        self.button_start_preprocessing = StyledButton("🚀 Start Extraction", "#27ae60", "#229954")
        self.button_start_preprocessing.setMinimumHeight(45)
        self.button_start_preprocessing.clicked.connect(self.start_preprocessing)
        
        self.back_button = StyledButton("← Back to Hub", "#95a5a6", "#7f8c8d")
        self.back_button.setMinimumHeight(45)
        self.back_button.clicked.connect(self.back_to_hub)
        
        buttons_layout.addWidget(self.button_start_preprocessing)
        buttons_layout.addWidget(self.back_button)
        
        main_layout.addLayout(buttons_layout)

    def browse_dataset(self):
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Select Music Dataset Directory")
        if self.dataset_path:
            self.line_edit_dataset.setText(self.dataset_path)
            self.status_label.setText("Dataset directory selected")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")

    def browse_output(self):
        self.output_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if self.output_path:
            self.line_edit_output.setText(self.output_path)
            self.status_label.setText("Output directory selected")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")

    def start_preprocessing(self):
        self.filename = self.line_edit_filename.text()
        if not self.dataset_path or not self.output_path or not self.filename:
            QMessageBox.warning(self, "Missing Information", "Please fill in all fields before starting.")
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.button_start_preprocessing.setEnabled(False)
        self.status_label.setText("Processing... Please wait")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold; font-size: 14px;")
        
        try:
            # Simulate progress (you can integrate this with your actual processing)
            QTimer.singleShot(100, self.process_mfcc)
        except Exception as e:
            self.show_error(f"Error during processing: {str(e)}")

    def process_mfcc(self):
        try:
            # Call the actual MFCC extraction
            MFCC_extraction.main(self.dataset_path, self.output_path, self.filename)
            
            # Success
            self.progress_bar.setVisible(False)
            self.status_label.setText("✅ Extraction completed successfully!")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")
            self.button_start_preprocessing.setEnabled(True)
            
            QMessageBox.information(self, "Success", "MFCC feature extraction completed successfully!")
            
        except Exception as e:
            self.show_error(f"Error during processing: {str(e)}")

    def show_error(self, message):
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Error occurred")
        self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 14px;")
        self.button_start_preprocessing.setEnabled(True)
        QMessageBox.critical(self, "Error", message)

    def back_to_hub(self):
        self.close()
        self.hub_window = HubWindow()
        self.hub_window.show()

class TrainModelWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenreDiscern - Model Training")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
        """)
        
        self.mfcc_path = ""
        self.model_type = ""
        self.output_directory = ""
        self.initial_lr_value = 0.0001
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_label = QLabel("🧠 Neural Network Model Training")
        header_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                padding: 15px;
                background-color: #ecf0f1;
                border-radius: 8px;
                border: 2px solid #bdc3c7;
            }
        """)
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel("Train deep learning models on extracted MFCC features for music genre classification.")
        desc_label.setStyleSheet("color: #34495e; font-size: 14px; padding: 10px;")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        main_layout.addWidget(desc_label)
        
        # Input form
        form_group = QGroupBox("⚙️ Training Configuration")
        form_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        form_layout = QGridLayout(form_group)
        form_layout.setSpacing(15)
        
        # MFCC path
        mfcc_label = QLabel("📊 MFCC Features File:")
        mfcc_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        form_layout.addWidget(mfcc_label, 0, 0)
        
        self.line_edit_mfcc = QLineEdit()
        self.line_edit_mfcc.setPlaceholderText("Select JSON file containing extracted MFCC features...")
        self.line_edit_mfcc.setReadOnly(True)
        self.line_edit_mfcc.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                font-size: 14px;
            }
        """)
        form_layout.addWidget(self.line_edit_mfcc, 0, 1)
        
        self.button_select_mfcc = StyledButton("Browse", "#3498db", "#2980b9")
        self.button_select_mfcc.clicked.connect(self.select_mfcc_path)
        form_layout.addWidget(self.button_select_mfcc, 0, 2)
        
        # Model type
        model_label = QLabel("🏗️ Model Architecture:")
        model_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        form_layout.addWidget(model_label, 1, 0)
        
        self.combo_model_type = QComboBox()
        self.combo_model_type.addItems(["FC", "CNN", "LSTM", "xLSTM", "GRU", "Tr_FC", "Tr_CNN", "Tr_LSTM", "Tr_GRU"])
        self.combo_model_type.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                font-size: 14px;
                min-width: 200px;
            }
        """)
        self.combo_model_type.currentTextChanged.connect(self.on_model_type_changed)
        form_layout.addWidget(self.combo_model_type, 1, 1)
        
        # Model description
        self.model_desc_label = QLabel("Select a model architecture to see description")
        self.model_desc_label.setStyleSheet("color: #7f8c8d; font-size: 12px; font-style: italic;")
        self.model_desc_label.setWordWrap(True)
        form_layout.addWidget(self.model_desc_label, 1, 2)
        
        # Output directory
        output_label = QLabel("📁 Output Directory:")
        output_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        form_layout.addWidget(output_label, 2, 0)
        
        self.line_edit_output = QLineEdit()
        self.line_edit_output.setPlaceholderText("Select directory to save trained model...")
        self.line_edit_output.setReadOnly(True)
        self.line_edit_output.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                font-size: 14px;
            }
        """)
        form_layout.addWidget(self.line_edit_output, 2, 1)
        
        self.button_select_output = StyledButton("Browse", "#3498db", "#2980b9")
        self.button_select_output.clicked.connect(self.select_output_directory)
        form_layout.addWidget(self.button_select_output, 2, 2)
        
        # Learning rate
        lr_label = QLabel("📈 Learning Rate:")
        lr_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        form_layout.addWidget(lr_label, 3, 0)
        
        self.slider_lr = QSlider(Qt.Horizontal)
        self.slider_lr.setRange(-6, -1)  # 10^-6 to 10^-1
        self.slider_lr.setValue(-4)  # Default to 10^-4
        self.slider_lr.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 8px;
                background: #ecf0f1;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #2980b9;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)
        self.slider_lr.valueChanged.connect(self.on_lr_changed)
        form_layout.addWidget(self.slider_lr, 3, 1)
        
        self.lr_value_label = QLabel("0.0001")
        self.lr_value_label.setStyleSheet("color: #2c3e50; font-weight: bold; font-size: 14px;")
        form_layout.addWidget(self.lr_value_label, 3, 2)
        
        # Batch size control
        batch_size_layout = QHBoxLayout()
        batch_size_label = QLabel("Batch Size:")
        batch_size_label.setStyleSheet("color: #34495e; font-size: 12px;")
        batch_size_layout.addWidget(batch_size_label)
        
        self.batch_size_combo = QComboBox()
        self.batch_size_combo.addItems(["4", "8", "16", "32", "64", "128"])
        self.batch_size_combo.setCurrentText("32")  # Default to 32 for better training efficiency
        self.batch_size_combo.setToolTip("""
        Batch Size Guide:
        • 4-8: Very low memory, slow training, good for limited GPU
        • 16-32: Low memory, moderate training speed, recommended for most cases
        • 64-128: Higher memory, fast training, requires sufficient GPU memory
        
        Smaller batches use less memory but train slower.
        """)
        self.batch_size_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                font-size: 12px;
                min-width: 80px;
            }
        """)
        batch_size_layout.addWidget(self.batch_size_combo)
        
        form_layout.addLayout(batch_size_layout, 4, 1)
        
        main_layout.addWidget(form_group)
        
        # Progress section
        progress_group = QGroupBox("⚡ Training Progress")
        progress_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        progress_layout = QVBoxLayout(progress_group)
        
        # Cute animated training indicator
        self.training_animation_label = QLabel("🎯 Ready to train")
        self.training_animation_label.setVisible(False)
        self.training_animation_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 48px;
                font-weight: bold;
                padding: 20px;
                text-align: center;
                background-color: #ecf0f1;
                border-radius: 15px;
                border: 3px solid #3498db;
                min-height: 80px;
            }
        """)
        self.training_animation_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.training_animation_label)
        
        self.status_label = QLabel("Ready to train")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        main_layout.addWidget(progress_group)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(20)
        
        self.button_train = StyledButton("🚀 Start Training", "#e74c3c", "#c0392b")
        self.button_train.setMinimumHeight(45)
        self.button_train.clicked.connect(self.start_training_clicked)
        
        self.back_button = StyledButton("← Back to Hub", "#95a5a6", "#7f8c8d")
        self.back_button.setMinimumHeight(45)
        self.back_button.clicked.connect(self.back_to_hub)
        
        buttons_layout.addWidget(self.button_train)
        buttons_layout.addWidget(self.back_button)
        
        main_layout.addLayout(buttons_layout)
        
        # Initialize model descriptions
        self.model_descriptions = {
            "FC": "Fully Connected Neural Network - Simple feedforward network",
            "CNN": "Convolutional Neural Network - Good for spatial feature extraction",
            "LSTM": "Long Short-Term Memory - Excellent for sequential data",
            "xLSTM": "Extended LSTM - Advanced LSTM with causal convolutions",
            "GRU": "Gated Recurrent Unit - Efficient alternative to LSTM",
            "Tr_FC": "Transformer with FC layers - Attention-based architecture",
            "Tr_CNN": "Transformer with CNN layers - Hybrid attention-convolution",
            "Tr_LSTM": "Transformer with LSTM layers - Sequential attention model",
            "Tr_GRU": "Transformer with GRU layers - Efficient attention model"
        }
        
        self.on_model_type_changed()
        
        # Initialize training worker
        self.training_worker = None
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_training_progress)
        
        # Animation frames for cute training indicator
        self.animation_frames = [
            "🚀 Training...",
            "🧠 Learning...", 
            "⚡ Processing...",
            "🎵 Analyzing...",
            "🔬 Computing...",
            "🌟 Optimizing...",
            "💫 Training...",
            "🎯 Learning..."
        ]
        self.current_frame = 0

    def select_mfcc_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSON File", "", "JSON Files (*.json)")
        if file_path:
            self.mfcc_path = file_path
            self.line_edit_mfcc.setText(file_path)
            self.status_label.setText("MFCC file selected")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")

    def on_model_type_changed(self):
        self.model_type = self.combo_model_type.currentText()
        self.model_desc_label.setText(self.model_descriptions.get(self.model_type, ""))

    def select_output_directory(self):
        self.output_directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if self.output_directory:
            self.line_edit_output.setText(self.output_directory)
            self.status_label.setText("Output directory selected")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")

    def on_lr_changed(self):
        self.initial_lr_value = 10 ** self.slider_lr.value()
        self.lr_value_label.setText(f"{self.initial_lr_value:.6f}")

    def start_training_clicked(self):
        """Handle start training button click"""
        batch_size = int(self.batch_size_combo.currentText())
        
        # Validate inputs before starting
        if not self.mfcc_path:
            QMessageBox.warning(self, "Missing Information", "Please select an MFCC features file.")
            return
        if not self.output_directory:
            QMessageBox.warning(self, "Missing Information", "Please select an output directory.")
            return
        if not self.model_type:
            QMessageBox.warning(self, "Missing Information", "Please select a model type.")
            return
        
        # Set up progress bar and status
        self.training_animation_label.setVisible(True)
        self.training_animation_label.setText(self.animation_frames[0])
        self.status_label.setText("Preparing training...")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold; font-size: 14px;")
        
        # Create and start training worker
        self.training_worker = TrainingWorker(
            self.mfcc_path, 
            self.model_type, 
            self.output_directory, 
            str(self.initial_lr_value), 
            batch_size
        )
        
        # Connect worker signals
        self.training_worker.training_started.connect(self.on_training_started)
        self.training_worker.progress_update.connect(self.on_progress_update)
        self.training_worker.training_finished.connect(self.on_training_finished)
        self.training_worker.training_error.connect(self.on_training_error)
        
        # Start the worker thread
        self.training_worker.start()

    def on_training_started(self):
        """Callback when training worker starts"""
        self.status_label.setText("Training model...")
        self.training_animation_label.setText(self.animation_frames[0])
        self.training_animation_label.setVisible(True)
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold; font-size: 14px;")
        self.button_train.setEnabled(False)
        
        # Start periodic progress updates to show activity
        self.progress_timer.start(800)  # Update every 800ms for smooth animation

    def on_progress_update(self, message):
        """Callback for progress updates from training worker"""
        self.status_label.setText(message)

    def on_training_finished(self, result):
        """Callback when training worker finishes"""
        self.cleanup_worker()
        self.status_label.setText("✅ Training completed successfully!")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")
        QMessageBox.information(self, "Success", "Model training completed successfully!")
        self.show_ready_state()

    def on_training_error(self, error_msg):
        """Callback for training errors"""
        self.cleanup_worker()
        self.status_label.setText("❌ Training failed")
        self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 14px;")
        QMessageBox.critical(self, "Training Error", f"An error occurred during training:\n\n{error_msg}")
        self.show_ready_state()

    def cleanup_worker(self):
        """Clean up training worker resources"""
        if self.training_worker:
            if self.training_worker.isRunning():
                self.training_worker.terminate()
                self.training_worker.wait()
            self.training_worker.deleteLater()
            self.training_worker = None
        
        # Stop progress timer
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()

    def closeEvent(self, event):
        """Handle window closing safely"""
        if self.training_worker and self.training_worker.isRunning():
            reply = QMessageBox.question(
                self, 
                "Training in Progress", 
                "Training is currently running. Are you sure you want to close the window?\n\nThis will stop the training process.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                # Stop the training worker
                self.training_worker.terminate()
                self.training_worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def update_training_status(self, epoch=None, batch=None, total_epochs=None, total_batches=None):
        """Update training status with actual epoch/batch information"""
        # Just update the status label with current training info
        if epoch is not None:
            message = f"Training: Epoch {epoch}"
            if batch is not None:
                message += f", Batch {batch}"
            self.status_label.setText(message)

    def update_training_progress(self):
        """Update the cute animation to show training activity"""
        if hasattr(self, 'training_animation_label') and self.training_animation_label.isVisible():
            # Cycle through animation frames
            self.current_frame = (self.current_frame + 1) % len(self.animation_frames)
            self.training_animation_label.setText(self.animation_frames[self.current_frame])

    def show_ready_state(self):
        """Show the ready state when not training"""
        self.training_animation_label.setText("🎯 Ready to train")
        self.training_animation_label.setVisible(True)
        self.status_label.setText("Ready to train")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")
        self.button_train.setEnabled(True)

    def show_error(self, message):
        self.show_ready_state()
        
        # Special handling for CUDA OOM errors
        if "out of memory" in message.lower() or "cuda" in message.lower():
            self.show_memory_error_dialog()
        else:
            QMessageBox.critical(self, "Error", message)

    def show_memory_error_dialog(self):
        """Show specialized dialog for memory-related errors"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("CUDA Out of Memory Error")
        msg.setText("Your GPU ran out of memory during training.")
        msg.setInformativeText("""
        To resolve this issue, try:
        
        1. Reduce batch size (use 16 or 32 instead of 64/128)
        2. Enable memory optimization
        3. Close other GPU applications
        4. Use a smaller model architecture
        5. Reduce sequence length if possible
        
        Would you like to try with a smaller batch size?
        """)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        
        if msg.exec_() == QMessageBox.Yes:
            # Automatically reduce batch size
            current_batch_size = int(self.batch_size_combo.currentText())
            if current_batch_size > 4:
                new_batch_size = str(current_batch_size // 2)
                self.batch_size_combo.setCurrentText(new_batch_size)
                self.status_label.setText(f"Batch size reduced to {new_batch_size}. Ready to retry.")
                self.status_label.setStyleSheet("color: #f39c12; font-weight: bold; font-size: 14px;")
            else:
                self.status_label.setText("Batch size already at minimum (4). Try memory optimization or smaller model.")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 14px;")

    def back_to_hub(self):
        self.close()
        self.hub_window = HubWindow()
        self.hub_window.show()

################################################################################################################################
################################################################################################################################

def main():
    app = QApplication(sys.argv)
    
    # Set application-wide styling
    app.setStyleSheet("""
        QApplication {
            font-family: 'Segoe UI', 'Arial', sans-serif;
        }
    """)
    
    welcome_window = WelcomeWindow()
    welcome_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
