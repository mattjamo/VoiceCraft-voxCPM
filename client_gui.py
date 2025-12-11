import sys
import os
import shutil
import requests
import tempfile
import pygame
import threading
import pyperclip
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QTextEdit, QPushButton, 
                             QSlider, QCheckBox, QFrame, QGraphicsDropShadowEffect, QComboBox,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QInputDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt6.QtGui import QColor
import sounddevice as sd
import soundfile as sf
import numpy as np

# Configuration
SERVER_URL = "http://localhost:5000/v1/audio/speech"
VOICES_URL = "http://localhost:5000/v1/voices"
PATHS_URL = "http://localhost:5000/v1/system/paths"
DEFAULT_FONT = "Segoe UI"
OUTPUT_DIR = os.path.abspath("saved_outputs")
TEMP_DIR = os.path.abspath("temp")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

class AudioWorker(QThread):
    finished = pyqtSignal(bool, str) # Success, Message/Path
    
    def __init__(self, text, cfg_value, timesteps, voice, retry_badcase, retry_max, retry_threshold, stream=False):
        super().__init__()
        self.text = text
        self.cfg_value = cfg_value
        self.timesteps = timesteps
        self.voice = voice
        self.retry_badcase = retry_badcase
        self.retry_max = retry_max
        self.retry_threshold = retry_threshold
        self.stream = stream

    def run(self):
        try:
            payload = {
                "input": self.text,
                "model": "voxcpm",
                "voice": self.voice,
                "cfg_value": self.cfg_value,
                "inference_timesteps": self.timesteps,
                "stream": self.stream,
                # Retry parameters
                "retry_badcase": self.retry_badcase,
                "retry_badcase_max_times": self.retry_max,
                "retry_badcase_ratio_threshold": self.retry_threshold
            }
            
            response = requests.post(SERVER_URL, json=payload, stream=self.stream)
            
            if response.status_code == 200:
                # Save to temp file
                fd, path = tempfile.mkstemp(suffix=".wav", dir=TEMP_DIR)
                os.close(fd) # Close file descriptor, sf.write opens it by name or handle
                
                if self.stream:
                    # Collect PCM chunks and convert to WAV
                    pcm_data = []
                    for chunk in response.iter_content(chunk_size=4096):
                        if chunk:
                            pcm_data.append(chunk)
                    
                    raw_bytes = b"".join(pcm_data)
                    # Convert raw bytes to numpy array
                    audio_array = np.frombuffer(raw_bytes, dtype=np.int16)
                    
                    # Write as WAV using soundfile
                    sf.write(path, audio_array, 44100) # VoxCPM 1.5 uses 44.1kHz.
                else:
                    with open(path, 'wb') as f:
                        f.write(response.content)

                self.finished.emit(True, path)
            else:
                self.finished.emit(False, f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            self.finished.emit(False, str(e))

class RecorderWorker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, filename, text, samplerate=44100):
        super().__init__()
        self.filename = filename
        self.text = text # Transcript to save
        self.samplerate = samplerate
        self.recording = []
        self.is_recording = True
        self.stream = None

    def run(self):
        try:
            # Callback for sounddevice
            def callback(indata, frames, time, status):
                if self.is_recording:
                    self.recording.append(indata.copy())
                else:
                    raise sd.CallbackStop

            with sd.InputStream(samplerate=self.samplerate, channels=1, callback=callback):
                while self.is_recording:
                    self.msleep(100)
            
            # Save Audio
            audio_data = np.concatenate(self.recording, axis=0)
            sf.write(self.filename, audio_data, self.samplerate)

            # Save Transcript
            txt_path = os.path.splitext(self.filename)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(self.text)

            self.finished.emit(True, self.filename)
            
        except Exception as e:
            self.finished.emit(False, str(e))

    def stop(self):
        self.is_recording = False
        
class ModernWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(800, 600)
        
        # Audio Init
        pygame.mixer.init()
        
        # Central Widget & Layout
        self.central_widget = QWidget()
        self.central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(self.central_widget)
        
        # Main Layout
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Background Frame (for styling and shadow)
        self.bg_frame = QFrame()
        self.bg_frame.setObjectName("BgFrame")
        self.main_layout.addWidget(self.bg_frame)
        
        self.bg_layout = QVBoxLayout(self.bg_frame)
        self.bg_layout.setContentsMargins(20, 20, 20, 20)
        
        # UI Components
        self.setup_ui()
        self.setup_styles()
        
        # Initial Data
        self.fetch_voices()
        
        # Drag Logic
        self.old_pos = None

        # Random Generation Flag
        self.is_random_generation = False

    def fetch_voices(self):
        try:
            # Try to fetch from server if running
            response = requests.get(VOICES_URL, timeout=1)
            if response.status_code == 200:
                voices = response.json().get("voices", ["default"])
                current = self.combo_voice.currentText()
                self.combo_voice.clear()
                self.combo_voice.addItems(voices)
                # Restore selection if exists
                if current in voices:
                    self.combo_voice.setCurrentText(current)
            else:
                if self.combo_voice.count() == 0: self.combo_voice.addItem("default")
        except:
             # Server might be down or starting, keep default
             if self.combo_voice.count() == 0: self.combo_voice.addItem("default")

    def setup_ui(self):
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("VoxCPM Client")
        title.setObjectName("TitleLabel")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        min_btn = QPushButton("_")
        min_btn.setObjectName("TitleBtn")
        min_btn.clicked.connect(self.showMinimized)
        header_layout.addWidget(min_btn)
        
        close_btn = QPushButton("X")
        close_btn.setObjectName("TitleBtn")
        close_btn.setObjectName("CloseBtn") # Overwrite for red hover
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)
        
        self.bg_layout.addLayout(header_layout)
        
        # Content Layout (Sidebar + Main)
        content_layout = QHBoxLayout()
        
        # Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(250)
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        
        # Controls Group
        lbl_cfg = QLabel("CFG Scale: 2.0")
        self.slider_cfg = QSlider(Qt.Orientation.Horizontal)
        self.slider_cfg.setRange(1, 100) # 0.1 - 10.0
        self.slider_cfg.setValue(20)
        self.slider_cfg.valueChanged.connect(lambda v: lbl_cfg.setText(f"CFG Scale: {v/10.0:.1f}"))
        sidebar_layout.addWidget(lbl_cfg)
        sidebar_layout.addWidget(self.slider_cfg)
        
        sidebar_layout.addSpacing(15)
        
        lbl_steps = QLabel("Inference Steps: 10")
        self.slider_steps = QSlider(Qt.Orientation.Horizontal)
        self.slider_steps.setRange(1, 50)
        self.slider_steps.setValue(10)
        self.slider_steps.valueChanged.connect(lambda v: lbl_steps.setText(f"Inference Steps: {v}"))
        sidebar_layout.addWidget(lbl_steps)
        sidebar_layout.addWidget(self.slider_steps)
        
        sidebar_layout.addSpacing(15)
        
        lbl_voice = QLabel("Voice Profile:")
        self.combo_voice = QComboBox()
        self.combo_voice.addItem("default")
        sidebar_layout.addWidget(lbl_voice)
        sidebar_layout.addWidget(self.combo_voice)
        
        btn_refresh = QPushButton("⟳ Refresh Voices")
        btn_refresh.setObjectName("SmallBtn")
        btn_refresh.clicked.connect(self.fetch_voices)
        sidebar_layout.addWidget(btn_refresh)
        
        sidebar_layout.addSpacing(20)
        
        # Advanced Group
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.setContentsMargins(5, 10, 5, 5)
        
        self.chk_retry = QCheckBox("Retry Bad Cases")
        self.chk_retry.setChecked(True)
        advanced_layout.addWidget(self.chk_retry)

        self.chk_stream = QCheckBox("Enable Streaming")
        self.chk_stream.setChecked(False)
        advanced_layout.addWidget(self.chk_stream)
        
        max_retry_layout = QHBoxLayout()
        max_retry_layout.addWidget(QLabel("Max Retries:"))
        self.spin_retry_max = QSpinBox()
        self.spin_retry_max.setRange(1, 10)
        self.spin_retry_max.setValue(3)
        max_retry_layout.addWidget(self.spin_retry_max)
        advanced_layout.addLayout(max_retry_layout)
        
        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("Ratio Thresh:"))
        self.spin_retry_thresh = QDoubleSpinBox()
        self.spin_retry_thresh.setRange(1.0, 20.0)
        self.spin_retry_thresh.setValue(6.0)
        self.spin_retry_thresh.setSingleStep(0.5)
        ratio_layout.addWidget(self.spin_retry_thresh)
        advanced_layout.addLayout(ratio_layout)
        
        sidebar_layout.addWidget(advanced_group)

        # System Group
        system_group = QGroupBox("System & Output")
        system_layout = QVBoxLayout(system_group)
        system_layout.setContentsMargins(5, 10, 5, 5)

        self.chk_autosave = QCheckBox("Autosave Audio")
        self.chk_autosave.setChecked(False) 
        system_layout.addWidget(self.chk_autosave)
        
        btn_open_saved = QPushButton("📂 Open Saved")
        btn_open_saved.setObjectName("SmallBtn")
        btn_open_saved.clicked.connect(self.open_saved_folder)
        system_layout.addWidget(btn_open_saved)

        btn_open_model = QPushButton("🧠 Open Model Dir")
        btn_open_model.setObjectName("SmallBtn")
        btn_open_model.clicked.connect(self.open_model_folder)
        system_layout.addWidget(btn_open_model)

        sidebar_layout.addWidget(system_group)
        

        
        # Recorder Group
        recorder_group = QGroupBox("Voice Cloning / Recorder / Random Voice Gen")
        rec_layout = QVBoxLayout(recorder_group)
        rec_layout.setContentsMargins(5, 10, 5, 5)
        
        from PyQt6.QtWidgets import QLineEdit
        self.txt_voice_name = QLineEdit()
        self.txt_voice_name.setPlaceholderText("Voice Name (e.g. my_voice)")
        self.txt_voice_name.setObjectName("TextInput") # Reuse style
        # Custom smaller height for sidebar input
        self.txt_voice_name.setStyleSheet("padding: 5px; font-size: 12px;")
        rec_layout.addWidget(self.txt_voice_name)
        
        self.btn_record = QPushButton("🔴 Record")
        self.btn_record.setObjectName("SmallBtn")
        self.btn_record.setStyleSheet("text-align: center; color: #ff5555; border-color: #ff5555;")
        self.btn_record.clicked.connect(self.toggle_recording)
        rec_layout.addWidget(self.btn_record)
        
        self.btn_random = QPushButton("🎲 Generate Random Voice")
        self.btn_random.setObjectName("SmallBtn")
        self.btn_random.clicked.connect(self.generate_random_voice)
        rec_layout.addWidget(self.btn_random)
        
        rec_label = QLabel("Read text in main box.")
        rec_label.setWordWrap(True)
        rec_label.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        rec_layout.addWidget(rec_label)
        
        sidebar_layout.addWidget(recorder_group)

        sidebar_layout.addStretch()
        content_layout.addWidget(sidebar)
        
        # Main Area
        main_area = QVBoxLayout()
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text here to generate speech...")
        self.text_input.setObjectName("TextInput")
        main_area.addWidget(self.text_input)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_paste = QPushButton("📋 Play Paste")
        self.btn_paste.setObjectName("ActionBtn")
        self.btn_paste.clicked.connect(self.play_paste)
        btn_layout.addWidget(self.btn_paste)
        
        self.btn_generate = QPushButton("▶ Generate")
        self.btn_generate.setObjectName("ActionBtn")
        self.btn_generate.clicked.connect(self.generate_and_play)
        btn_layout.addWidget(self.btn_generate)
        
        main_area.addLayout(btn_layout)
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setObjectName("StatusLabel")
        main_area.addWidget(self.status_label)
        
        content_layout.addLayout(main_area)
        self.bg_layout.addLayout(content_layout)

    def setup_styles(self):
        self.setStyleSheet("""
            QWidget { font-family: 'Segoe UI', sans-serif; color: #E0E0E0; }
            
            #BgFrame {
                background-color: rgba(30, 30, 30, 240); 
                border-radius: 15px;
                border: 1px solid #404040;
            }
            
            #TitleLabel { font-size: 18px; font-weight: bold; color: #FFFFFF; }
            
            QPushButton#TitleBtn {
                background: transparent; border: none; font-size: 16px; width: 30px;
            }
            QPushButton#TitleBtn:hover { background-color: #404040; border-radius: 5px; }
            QPushButton#CloseBtn:hover { background-color: #C42B1C; color: white; border-radius: 5px; }
            
            #Sidebar { border-right: 1px solid #404040; margin-right: 10px; }
            
            QGroupBox {
                border: 1px solid #404040; border-radius: 5px; margin-top: 10px; padding-top: 10px; font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            
            QSlider::groove:horizontal {
                border: 1px solid #3A3939; height: 8px; background: #201F1F; margin: 2px 0; border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #007ACC; border: 1px solid #007ACC; width: 18px; margin: -2px 0; border-radius: 9px;
            }
            
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #252526; border: 1px solid #3E3E42; border-radius: 5px; padding: 5px; color: #F0F0F0;
            }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; border-left: 1px solid #3E3E42; width: 10px; }
            
            
            QPushButton#SmallBtn {
                background-color: transparent; border: 1px solid #404040; border-radius: 4px; padding: 4px; color: #CCCCCC;
            }
            QPushButton#SmallBtn:hover { background-color: #333333; }
            
            #TextInput {
                background-color: #252526; border: 1px solid #3E3E42; border-radius: 8px;
                font-size: 14px; padding: 10px; color: #F0F0F0; selection-background-color: #264F78;
            }
            
            QPushButton#ActionBtn {
                background-color: #0E639C; color: white; border-radius: 6px; 
                padding: 10px 20px; font-size: 14px; font-weight: bold;
            }
            QPushButton#ActionBtn:hover { background-color: #1177BB; }
            QPushButton#ActionBtn:pressed { background-color: #0D5685; }
            
            #StatusLabel { color: #808080; margin-top: 5px; }
        """)
        
        # Shadow Effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 0)
        self.bg_frame.setGraphicsEffect(shadow)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.old_pos is not None:
            delta = QPoint(event.globalPosition().toPoint() - self.old_pos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.old_pos = None

    def play_paste(self):
        text = pyperclip.paste()
        if text:
            self.text_input.setText(text)
            self.generate_and_play()
        else:
            self.status_label.setText("Clipboard empty!")

    def generate_random_voice(self):
        self.is_random_generation = True
        self.generate_and_play(voice_override="default")

    def generate_and_play(self, voice_override=None):
        text = self.text_input.toPlainText().strip()
        if not text:
            self.status_label.setText("Please enter text!")
            return
            
        self.status_label.setText("Generating...")
        self.btn_generate.setEnabled(False)
        self.btn_paste.setEnabled(False)
        
        cfg = self.slider_cfg.value() / 10.0
        steps = self.slider_steps.value()
        voice = voice_override if voice_override else self.combo_voice.currentText()
        
        retry_bad = self.chk_retry.isChecked()
        retry_max = self.spin_retry_max.value()
        retry_thresh = self.spin_retry_thresh.value()
        stream = self.chk_stream.isChecked()
        
        self.worker = AudioWorker(text, cfg, steps, voice, retry_bad, retry_max, retry_thresh, stream)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.start()

    def on_generation_finished(self, success, result):
        self.btn_generate.setEnabled(True)
        self.btn_paste.setEnabled(True)
        
        if success:
            self.status_label.setText("Playing...")
            try:
                # Play audio
                pygame.mixer.music.load(result)
                pygame.mixer.music.play()
                
                # Autosave Logic
                if self.chk_autosave.isChecked() and not self.is_random_generation:
                    try:
                        import time
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"{self.worker.voice}_{timestamp}.wav"
                        dest = os.path.join(OUTPUT_DIR, filename)
                        shutil.copy2(result, dest)
                        self.status_label.setText(f"Playing... (Saved to {filename})")
                    except Exception as e:
                         print(f"Autosave failed: {e}")

                # Random Voice Save/Delete Dialog
                if self.is_random_generation:
                    text, ok = QInputDialog.getText(self, "Save Voice", "Enter a name for this random voice (Cancel to delete):")
                    if ok and text:
                        # Save
                        try:
                            voices_dir = os.path.abspath("voices")
                            if not os.path.exists(voices_dir):
                                os.makedirs(voices_dir)
                            
                            new_wav = os.path.join(voices_dir, f"{text}.wav")
                            new_txt = os.path.join(voices_dir, f"{text}.txt")
                            
                            shutil.copy2(result, new_wav)
                            with open(new_txt, "w", encoding="utf-8") as f:
                                f.write(self.text_input.toPlainText().strip())
                                
                            self.fetch_voices()
                            index = self.combo_voice.findText(text)
                            if index >= 0:
                                self.combo_voice.setCurrentIndex(index)
                                
                            self.status_label.setText(f"Voice '{text}' saved!")
                        except Exception as e:
                            self.status_label.setText(f"Error saving voice: {e}")
                            
                        # Cleanup temp
                        try:
                            # We might need to stop playback before deleting if using windows?
                            # pygame loads into memory? No, usually locks file.
                            # But we are using music.load. 
                            # Let's try to just remove it. If it fails, we might need to unload.
                            pass 
                        except:
                           pass
                    else:
                        # Delete / Cancel
                        self.status_label.setText("Random voice discarded.")
                    
                    # Always cleanup temp file for random gen
                    # Wait for playback to finish? Or just queue deletion?
                    # pygame.mixer.music.load/play locks the file on Windows.
                    # We can't delete it while it's playing.
                    # IMPORTANT: effectively we can't fully delete it immediately if it's playing.
                    # But the user might want to listen to decide.
                    # Solution: We can't delete immediately. We can tag it for deletion or just leave it since it's in temp.
                    # User asked to cleanup.
                    # We can stop playback if they hit Cancel?
                    if not ok:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload() # available in pygame 2.0+
                        try:
                           os.remove(result)
                        except Exception as e:
                           print(f"Cleanup failed: {e}")
                    else:
                         # If saved, we also want to clean up the temp file after we copied it.
                         # But it is playing.
                         pygame.mixer.music.stop()
                         pygame.mixer.music.unload()
                         try:
                           os.remove(result)
                           # Re-play from the saved file?
                           pygame.mixer.music.load(new_wav)
                           pygame.mixer.music.play()
                         except Exception as e:
                           print(f"Cleanup saved failed: {e}")
                    
                    self.is_random_generation = False


            except Exception as e:
                 self.status_label.setText(f"Playback Error: {e}")
        else:
            self.status_label.setText(f"Error: {result}")

    def open_saved_folder(self):
        try:
            os.startfile(OUTPUT_DIR)
        except Exception as e:
            self.status_label.setText(f"Error opening folder: {e}")

    def open_model_folder(self):
        try:
            response = requests.get(PATHS_URL, timeout=1)
            if response.status_code == 200:
                path = response.json().get("model_path")
                if path and os.path.exists(path):
                    os.startfile(path)
                else:
                    self.status_label.setText("Model path not found.")
            else:
                self.status_label.setText("Could not get model path from server.")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")


    def toggle_recording(self):
        if hasattr(self, 'recorder') and self.recorder and self.recorder.isRunning():
            # Stop Recording
            self.recorder.stop()
            self.btn_record.setText("🔴 Record")
            self.btn_record.setStyleSheet("text-align: center; color: #ff5555; border-color: #ff5555;")
            self.status_label.setText("Stopping recording...")
        else:
            # Start Recording
            name = self.txt_voice_name.text().strip()
            if not name:
                self.status_label.setText("Enter Voice Name first!")
                return
            
            text = self.text_input.toPlainText().strip()
            if not text:
                self.status_label.setText("Enter transcript text to read!")
                return
                
            # Define path
            voices_dir = os.path.abspath("voices")
            if not os.path.exists(voices_dir):
                os.makedirs(voices_dir)
                
            filename = os.path.join(voices_dir, f"{name}.wav")
            
            self.recorder = RecorderWorker(filename, text)
            self.recorder.finished.connect(self.on_recording_finished)
            self.recorder.start()
            
            self.btn_record.setText("⬛ Stop")
            self.btn_record.setStyleSheet("text-align: center; color: #ffffff; background-color: #cc0000; border-color: #cc0000;")
            self.status_label.setText(f"Recording '{name}'... Press Stop when done.")

    def on_recording_finished(self, success, result):
        if success:
            self.status_label.setText(f"Saved Voice: {os.path.basename(result)}")
            self.fetch_voices() # Refresh dropdown
            # Auto-select new voice
            name = os.path.splitext(os.path.basename(result))[0]
            index = self.combo_voice.findText(name)
            if index >= 0:
                self.combo_voice.setCurrentIndex(index)
        else:
            self.status_label.setText(f"Recording Error: {result}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernWindow()
    window.show()
    sys.exit(app.exec())
