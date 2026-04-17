"""
AcoustiTrack Pro - Acoustic Distance Measurement System
Clean and professional interface
"""

import sys
import numpy as np
import sounddevice as sd
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import queue
import time
import json
from datetime import datetime
from collections import deque


class LivePlotWidget(QWidget):
    """Small Qt-native plot widget used to avoid Matplotlib Qt backend crashes."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(500)
        self.series = [
            {
                'title': 'Distance Over Time',
                'ylabel': 'Distance (m)',
                'color': QColor('#2563eb'),
                'values': []
            },
            {
                'title': 'Confidence Level',
                'ylabel': 'Confidence (%)',
                'color': QColor('#16a34a'),
                'values': [],
                'fixed_range': (0, 105)
            },
            {
                'title': 'Sound Intensity',
                'ylabel': 'Intensity (dB)',
                'color': QColor('#dc2626'),
                'values': []
            }
        ]
        self.times = []
    
    def set_data(self, times, distances, confidences, intensities):
        self.times = list(times)
        values = [distances, confidences, intensities]
        for item, data in zip(self.series, values):
            item['values'] = list(data)
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('white'))
        
        margin_left = 62
        margin_right = 20
        margin_top = 18
        margin_bottom = 32
        gap = 18
        plot_count = len(self.series)
        plot_height = max(90, (self.height() - margin_top - margin_bottom - gap * (plot_count - 1)) // plot_count)
        
        for index, item in enumerate(self.series):
            top = margin_top + index * (plot_height + gap)
            rect = QRect(margin_left, top, self.width() - margin_left - margin_right, plot_height)
            self._draw_plot(painter, rect, item)
    
    def _draw_plot(self, painter, rect, item):
        painter.setPen(QPen(QColor('#d1d5db'), 1))
        painter.setBrush(QColor('#f8f9fa'))
        painter.drawRect(rect)
        
        painter.setPen(QPen(QColor('#111827'), 1))
        title_font = QFont(self.font())
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(rect.adjusted(0, -18, 0, 0), Qt.AlignTop | Qt.AlignHCenter, item['title'])
        
        painter.setFont(self.font())
        painter.setPen(QPen(QColor('#4b5563'), 1))
        painter.drawText(6, rect.center().y() + 5, item['ylabel'])
        painter.drawText(rect.left(), rect.bottom() + 18, 'Time (s)')
        
        for step in range(1, 4):
            y = rect.top() + step * rect.height() // 4
            painter.setPen(QPen(QColor('#e5e7eb'), 1))
            painter.drawLine(rect.left(), y, rect.right(), y)
        
        values = item['values']
        if len(values) < 2 or len(self.times) < 2:
            painter.setPen(QPen(QColor('#6b7280'), 1))
            painter.drawText(rect, Qt.AlignCenter, 'Waiting for measurement data')
            return
        
        min_time = min(self.times)
        max_time = max(self.times)
        if max_time <= min_time:
            max_time = min_time + 1
        
        if 'fixed_range' in item:
            min_value, max_value = item['fixed_range']
        else:
            min_value = min(values)
            max_value = max(values)
            if max_value <= min_value:
                max_value = min_value + 1
            padding = (max_value - min_value) * 0.1
            min_value -= padding
            max_value += padding
        
        points = []
        for t, value in zip(self.times, values):
            x_ratio = (t - min_time) / (max_time - min_time)
            y_ratio = (value - min_value) / (max_value - min_value)
            x = rect.left() + x_ratio * rect.width()
            y = rect.bottom() - y_ratio * rect.height()
            points.append(QPointF(x, y))
        
        painter.setPen(QPen(item['color'], 2))
        for start, end in zip(points, points[1:]):
            painter.drawLine(start, end)


# ==================== DISTANCE CALCULATOR ====================

class DistanceCalculator:
    """Improved distance calculation"""
    
    def __init__(self):
        self.calibration_points = []
        self.sample_rate = 44100
        
    def add_calibration(self, audio_data, distance):
        """Add calibration point"""
        features = self.extract_features(audio_data)
        self.calibration_points.append({
            'distance': distance,
            'features': features
        })
        print(f"✓ Calibration at {distance}m - Intensity: {features['intensity_db']:.1f} dB")
    
    def extract_features(self, audio_data):
        """Extract audio features"""
        rms = np.sqrt(np.mean(audio_data**2))
        intensity_db = 20 * np.log10(rms + 1e-10)
        peak = np.max(np.abs(audio_data))
        
        # FFT analysis
        fft_vals = np.abs(fft(audio_data))
        freqs = fftfreq(len(audio_data), 1/self.sample_rate)
        
        positive_fft = fft_vals[:len(fft_vals)//2]
        positive_freqs = freqs[:len(freqs)//2]
        
        low_freq = np.sum(positive_fft[(positive_freqs > 100) & (positive_freqs < 500)])
        high_freq = np.sum(positive_fft[(positive_freqs > 2000) & (positive_freqs < 8000)])
        
        hf_lf_ratio = high_freq / (low_freq + 1e-10)
        
        return {
            'rms': rms,
            'intensity_db': intensity_db,
            'peak': peak,
            'hf_lf_ratio': hf_lf_ratio
        }
    
    def estimate_distance(self, audio_data):
        """Estimate distance"""
        if len(self.calibration_points) == 0:
            return None, 0
        
        current_features = self.extract_features(audio_data)
        
        estimates = []
        confidences = []
        
        for ref in self.calibration_points:
            dist, conf = self._calculate_from_reference(current_features, ref)
            if dist is not None:
                estimates.append(dist)
                confidences.append(conf)
        
        if not estimates:
            return None, 0
        
        if sum(confidences) > 0:
            weights = np.array(confidences) / sum(confidences)
            final_distance = np.average(estimates, weights=weights)
            final_confidence = np.mean(confidences)
        else:
            final_distance = np.mean(estimates)
            final_confidence = 50
        
        return final_distance, final_confidence
    
    def _calculate_from_reference(self, current, reference):
        """Calculate distance from reference"""
        ref_features = reference['features']
        ref_distance = reference['distance']
        
        db_diff = current['intensity_db'] - ref_features['intensity_db']
        distance_ratio = 10 ** (-db_diff / 20)
        intensity_distance = ref_distance * distance_ratio
        
        if ref_features['hf_lf_ratio'] > 0 and current['hf_lf_ratio'] > 0:
            hf_ratio = current['hf_lf_ratio'] / ref_features['hf_lf_ratio']
            freq_distance = ref_distance / (hf_ratio ** 0.3)
        else:
            freq_distance = intensity_distance
        
        if ref_features['peak'] > 0 and current['peak'] > 0:
            peak_ratio = ref_features['peak'] / current['peak']
            peak_distance = ref_distance * np.sqrt(peak_ratio)
        else:
            peak_distance = intensity_distance
        
        final_distance = (0.6 * intensity_distance + 
                         0.25 * freq_distance + 
                         0.15 * peak_distance)
        
        final_distance = max(0.05, min(final_distance, 50.0))
        
        signal_strength = current['intensity_db']
        if signal_strength > -30:
            confidence = 95
        elif signal_strength > -40:
            confidence = 80
        elif signal_strength > -50:
            confidence = 60
        else:
            confidence = 30
        
        return final_distance, confidence
    
    def clear_calibration(self):
        """Clear calibrations"""
        self.calibration_points.clear()


# ==================== AUDIO PROCESSOR ====================

class AudioProcessor(QThread):
    """Audio capture thread"""
    audio_ready = pyqtSignal(np.ndarray, float)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.sample_rate = 44100
        self.block_size = 8192
        self.audio_queue = queue.Queue()
        
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback"""
        self.audio_queue.put(indata.copy())
    
    def run(self):
        """Main loop"""
        self.running = True
        
        with sd.InputStream(callback=self.audio_callback,
                          channels=1,
                          samplerate=self.sample_rate,
                          blocksize=self.block_size):
            
            while self.running:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    audio_flat = audio_data.flatten()
                    
                    # Bandpass filter
                    nyquist = self.sample_rate / 2
                    low = 100 / nyquist
                    high = 8000 / nyquist
                    b, a = butter(4, [low, high], btype='band')
                    filtered = filtfilt(b, a, audio_flat)
                    
                    # Calculate dB
                    rms = np.sqrt(np.mean(filtered**2))
                    db = 20 * np.log10(rms + 1e-10)
                    
                    self.audio_ready.emit(filtered, db)
                
                time.sleep(0.01)
    
    def stop(self):
        """Stop thread"""
        self.running = False


# ==================== MAIN APPLICATION ====================

class AcoustiTrackPro(QMainWindow):
    """Main application"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AcoustiTrack Pro - Acoustic Distance Measurement")
        self.setGeometry(100, 100, 1400, 850)
        
        self.distance_calc = DistanceCalculator()
        self.audio_processor = None
        
        self.distance_history = deque(maxlen=200)
        self.time_history = deque(maxlen=200)
        self.confidence_history = deque(maxlen=200)
        self.intensity_history = deque(maxlen=200)
        self.start_time = time.time()
        
        self.is_monitoring = False
        self.is_calibrated = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("AcoustiTrack Pro")
        header.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                padding: 15px;
                background-color: #ecf0f1;
                border-radius: 8px;
            }
        """)
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        # Content
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel
        left_panel = self.create_left_panel()
        content_splitter.addWidget(left_panel)
        
        # Right panel
        right_panel = self.create_right_panel()
        content_splitter.addWidget(right_panel)
        
        content_splitter.setStretchFactor(0, 2)
        content_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(content_splitter)
        
        # Apply clean theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                margin-top: 10px;
                padding: 15px;
                font-weight: bold;
                font-size: 13px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 500;
                min-height: 35px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
            QLineEdit {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 8px;
                background: white;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #007bff;
            }
        """)
    
    def create_left_panel(self):
        """Create left panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Controls
        controls = self.create_controls()
        layout.addWidget(controls)
        
        # Visualization
        viz = self.create_visualization()
        layout.addWidget(viz)
        
        return panel
    
    def create_controls(self):
        """Create control panel"""
        group = QGroupBox("Controls")
        layout = QVBoxLayout(group)
        
        # Calibration section
        cal_layout = QHBoxLayout()
        
        cal_layout.addWidget(QLabel("Distance (m):"))
        
        self.cal_input = QLineEdit("1.0")
        self.cal_input.setMaximumWidth(100)
        cal_layout.addWidget(self.cal_input)
        
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self.calibrate_system)
        cal_layout.addWidget(self.calibrate_btn)
        
        self.add_cal_btn = QPushButton("Add Point")
        self.add_cal_btn.clicked.connect(self.add_calibration_point)
        self.add_cal_btn.setEnabled(False)
        cal_layout.addWidget(self.add_cal_btn)
        
        self.clear_cal_btn = QPushButton("Clear")
        self.clear_cal_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.clear_cal_btn.clicked.connect(self.clear_calibrations)
        self.clear_cal_btn.setEnabled(False)
        cal_layout.addWidget(self.clear_cal_btn)
        
        cal_layout.addStretch()
        layout.addLayout(cal_layout)
        
        # Status
        self.cal_status = QLabel("Not calibrated")
        self.cal_status.setStyleSheet("""
            background-color: #fff3cd;
            color: #856404;
            padding: 10px;
            border-radius: 4px;
            font-weight: normal;
        """)
        layout.addWidget(self.cal_status)
        
        # Monitoring controls
        monitor_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.start_btn.clicked.connect(self.start_monitoring)
        self.start_btn.setEnabled(False)
        monitor_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        monitor_layout.addWidget(self.stop_btn)
        
        layout.addLayout(monitor_layout)
        
        # Data export
        export_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Data")
        save_btn.clicked.connect(self.save_data)
        export_layout.addWidget(save_btn)
        
        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(self.export_report)
        export_layout.addWidget(export_btn)
        
        layout.addLayout(export_layout)
        
        return group
    
    def create_visualization(self):
        """Create visualization"""
        group = QGroupBox("Live Graphs")
        layout = QVBoxLayout(group)
        
        self.canvas = LivePlotWidget()
        layout.addWidget(self.canvas)
        
        return group
    
    def create_right_panel(self):
        """Create right panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Readings
        readings = self.create_readings()
        layout.addWidget(readings)
        
        # Statistics
        stats = self.create_statistics()
        layout.addWidget(stats)
        
        # Info
        info = self.create_info()
        layout.addWidget(info)
        
        layout.addStretch()
        
        return panel
    
    def create_readings(self):
        """Create readings panel"""
        group = QGroupBox("Current Readings")
        layout = QVBoxLayout(group)
        
        self.distance_display = QLabel("---")
        self.distance_display.setStyleSheet("""
            font-size: 48px;
            font-weight: bold;
            color: #007bff;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 6px;
        """)
        self.distance_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.distance_display)
        
        unit = QLabel("meters")
        unit.setStyleSheet("font-size: 14px; color: #6c757d;")
        unit.setAlignment(Qt.AlignCenter)
        layout.addWidget(unit)
        
        self.confidence_display = QLabel("Confidence: ---%")
        self.confidence_display.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #28a745;
            padding: 8px;
            background-color: #d4edda;
            border-radius: 4px;
            margin-top: 10px;
        """)
        self.confidence_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_display)
        
        self.intensity_display = QLabel("Intensity: --- dB")
        self.intensity_display.setStyleSheet("""
            font-size: 14px;
            color: #dc3545;
            padding: 8px;
            background-color: #f8d7da;
            border-radius: 4px;
            margin-top: 5px;
        """)
        self.intensity_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.intensity_display)
        
        return group
    
    def create_statistics(self):
        """Create statistics panel"""
        group = QGroupBox("Statistics")
        layout = QVBoxLayout(group)
        
        label_style = "padding: 6px; font-size: 12px; font-weight: normal;"
        
        self.avg_label = QLabel("Average: --- m")
        self.avg_label.setStyleSheet(label_style)
        layout.addWidget(self.avg_label)
        
        self.min_label = QLabel("Minimum: --- m")
        self.min_label.setStyleSheet(label_style)
        layout.addWidget(self.min_label)
        
        self.max_label = QLabel("Maximum: --- m")
        self.max_label.setStyleSheet(label_style)
        layout.addWidget(self.max_label)
        
        self.std_label = QLabel("Std Dev: --- m")
        self.std_label.setStyleSheet(label_style)
        layout.addWidget(self.std_label)
        
        self.samples_label = QLabel("Samples: 0")
        self.samples_label.setStyleSheet(label_style)
        layout.addWidget(self.samples_label)
        
        return group
    
    def create_info(self):
        """Create info panel"""
        group = QGroupBox("System Info")
        layout = QVBoxLayout(group)
        
        info_style = "padding: 5px; font-size: 12px; font-weight: normal;"
        
        self.calibrations_label = QLabel("Calibrations: 0")
        self.calibrations_label.setStyleSheet(info_style)
        layout.addWidget(self.calibrations_label)
        
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet(info_style)
        layout.addWidget(self.status_label)
        
        return group
    
    def calibrate_system(self):
        """Calibrate system"""
        try:
            distance = float(self.cal_input.text())
            if distance <= 0:
                QMessageBox.warning(self, "Error", "Distance must be positive!")
                return
            
            self.distance_calc.clear_calibration()
            
            msg = QMessageBox(self)
            msg.setWindowTitle("Calibration")
            msg.setText(f"Calibrating at {distance} meters")
            msg.setInformativeText("Place sound source at specified distance.\nClick OK, then make a loud sound for 3 seconds.")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            
            if msg.exec_() == QMessageBox.Cancel:
                return
            
            duration = 3
            sample_rate = 44100
            
            progress = QProgressDialog("Recording...", None, 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            
            audio_data = sd.rec(int(duration * sample_rate), 
                              samplerate=sample_rate, 
                              channels=1)
            
            for i in range(100):
                time.sleep(duration / 100)
                progress.setValue(i + 1)
                QApplication.processEvents()
            
            sd.wait()
            progress.close()
            
            audio_flat = audio_data.flatten()
            self.distance_calc.add_calibration(audio_flat, distance)
            
            self.is_calibrated = True
            self.start_btn.setEnabled(True)
            self.add_cal_btn.setEnabled(True)
            self.clear_cal_btn.setEnabled(True)
            
            self.cal_status.setText(f"Calibrated at {distance}m")
            self.cal_status.setStyleSheet("""
                background-color: #d4edda;
                color: #155724;
                padding: 10px;
                border-radius: 4px;
                font-weight: normal;
            """)
            
            self.calibrations_label.setText(f"Calibrations: {len(self.distance_calc.calibration_points)}")
            
            QMessageBox.information(self, "Success", f"Calibrated at {distance}m!")
            
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid distance value!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calibration failed: {str(e)}")
    
    def add_calibration_point(self):
        """Add calibration point"""
        try:
            distance = float(self.cal_input.text())
            if distance <= 0:
                QMessageBox.warning(self, "Error", "Distance must be positive!")
                return
            
            msg = QMessageBox(self)
            msg.setWindowTitle("Add Calibration")
            msg.setText(f"Adding calibration at {distance} meters")
            msg.setInformativeText("Move sound source to new distance.\nClick OK, then make the same sound.")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            
            if msg.exec_() == QMessageBox.Cancel:
                return
            
            duration = 3
            sample_rate = 44100
            
            progress = QProgressDialog("Recording...", None, 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            
            audio_data = sd.rec(int(duration * sample_rate), 
                              samplerate=sample_rate, 
                              channels=1)
            
            for i in range(100):
                time.sleep(duration / 100)
                progress.setValue(i + 1)
                QApplication.processEvents()
            
            sd.wait()
            progress.close()
            
            audio_flat = audio_data.flatten()
            self.distance_calc.add_calibration(audio_flat, distance)
            
            self.calibrations_label.setText(f"Calibrations: {len(self.distance_calc.calibration_points)}")
            
            QMessageBox.information(self, "Success", f"Added calibration at {distance}m!")
            
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid distance value!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed: {str(e)}")
    
    def clear_calibrations(self):
        """Clear calibrations"""
        reply = QMessageBox.question(self, "Clear", 
                                    "Clear all calibrations?",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.distance_calc.clear_calibration()
            self.is_calibrated = False
            self.start_btn.setEnabled(False)
            self.add_cal_btn.setEnabled(False)
            self.clear_cal_btn.setEnabled(False)
            
            self.cal_status.setText("Not calibrated")
            self.cal_status.setStyleSheet("""
                background-color: #fff3cd;
                color: #856404;
                padding: 10px;
                border-radius: 4px;
                font-weight: normal;
            """)
            
            self.calibrations_label.setText("Calibrations: 0")
    
    def start_monitoring(self):
        """Start monitoring"""
        if not self.is_calibrated:
            QMessageBox.warning(self, "Error", "Please calibrate first!")
            return
        
        self.is_monitoring = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.calibrate_btn.setEnabled(False)
        self.add_cal_btn.setEnabled(False)
        
        self.distance_history.clear()
        self.time_history.clear()
        self.confidence_history.clear()
        self.intensity_history.clear()
        self.start_time = time.time()
        
        self.audio_processor = AudioProcessor()
        self.audio_processor.audio_ready.connect(self.process_audio)
        self.audio_processor.start()
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)
        
        self.status_label.setText("Status: Monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.calibrate_btn.setEnabled(True)
        self.add_cal_btn.setEnabled(True)
        
        if self.audio_processor:
            self.audio_processor.stop()
            self.audio_processor.wait()
        
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        
        self.status_label.setText("Status: Idle")
    
    def process_audio(self, audio_data, db):
        """Process audio"""
        if not self.is_monitoring:
            return
        
        distance, confidence = self.distance_calc.estimate_distance(audio_data)
        
        if distance is not None:
            current_time = time.time() - self.start_time
            
            self.distance_history.append(distance)
            self.time_history.append(current_time)
            self.confidence_history.append(confidence)
            self.intensity_history.append(db)
    
    def update_display(self):
        """Update display"""
        if not self.is_monitoring or len(self.distance_history) == 0:
            return
        
        # Current readings
        current_distance = self.distance_history[-1]
        current_confidence = self.confidence_history[-1]
        current_intensity = self.intensity_history[-1]
        
        self.distance_display.setText(f"{current_distance:.2f}")
        self.confidence_display.setText(f"Confidence: {current_confidence:.0f}%")
        self.intensity_display.setText(f"Intensity: {current_intensity:.1f} dB")
        
        # Statistics
        distances = list(self.distance_history)
        self.avg_label.setText(f"Average: {np.mean(distances):.2f} m")
        self.min_label.setText(f"Minimum: {np.min(distances):.2f} m")
        self.max_label.setText(f"Maximum: {np.max(distances):.2f} m")
        self.std_label.setText(f"Std Dev: {np.std(distances):.3f} m")
        self.samples_label.setText(f"Samples: {len(distances)}")
        
        # Update plots
        if len(self.time_history) > 1:
            times = list(self.time_history)
            distances = list(self.distance_history)
            confidences = list(self.confidence_history)
            intensities = list(self.intensity_history)
            
            self.canvas.set_data(times, distances, confidences, intensities)
    
    def save_data(self):
        """Save data"""
        if len(self.distance_history) == 0:
            QMessageBox.warning(self, "No Data", "No data to save!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "JSON Files (*.json)")
        if filename:
            data = {
                'timestamp': datetime.now().isoformat(),
                'calibrations': len(self.distance_calc.calibration_points),
                'measurements': [
                    {'time': t, 'distance': d, 'confidence': c, 'intensity': i}
                    for t, d, c, i in zip(self.time_history, self.distance_history, 
                                         self.confidence_history, self.intensity_history)
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            QMessageBox.information(self, "Success", "Data saved!")
    
    def export_report(self):
        """Export report"""
        if len(self.distance_history) == 0:
            QMessageBox.warning(self, "No Data", "No data to export!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, "Export Report", "", "Text Files (*.txt)")
        if filename:
            distances = list(self.distance_history)
            
            with open(filename, 'w') as f:
                f.write("ACOUSTITRACK PRO - MEASUREMENT REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Calibrations: {len(self.distance_calc.calibration_points)}\n")
                f.write(f"Samples: {len(distances)}\n\n")
                
                f.write("STATISTICS:\n")
                f.write(f"Average: {np.mean(distances):.3f} m\n")
                f.write(f"Minimum: {np.min(distances):.3f} m\n")
                f.write(f"Maximum: {np.max(distances):.3f} m\n")
                f.write(f"Std Dev: {np.std(distances):.3f} m\n\n")
                
                f.write("MEASUREMENTS:\n")
                f.write(f"{'Time(s)':<12} {'Distance(m)':<15} {'Confidence(%)':<15} {'Intensity(dB)':<15}\n")
                f.write("-" * 60 + "\n")
                
                for t, d, c, i in zip(self.time_history, self.distance_history, 
                                     self.confidence_history, self.intensity_history):
                    f.write(f"{t:<12.2f} {d:<15.3f} {c:<15.0f} {i:<15.1f}\n")
            
            QMessageBox.information(self, "Success", "Report exported!")
    
    def closeEvent(self, event):
        """Handle close"""
        if self.is_monitoring:
            self.stop_monitoring()
        event.accept()


# ==================== MAIN ====================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    print("=" * 60)
    print("AcoustiTrack Pro - Acoustic Distance Measurement")
    print("=" * 60)
    print("Starting application...\n")
    
    window = AcoustiTrackPro()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
