import sys
import speech_recognition as sr
import json
import time
import requests
import base64
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QListWidget, QTextEdit, QMessageBox, QInputDialog, QFileDialog, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
import sqlite3
import os
from mistralai import Mistral

# Класс Worker для обработки запросов к API Mistral в отдельном потоке
class Worker(QThread):
    result = pyqtSignal(str)

    def __init__(self, api_key, model, message):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.message = message

    def run(self):
        # Инициализация клиента Mistral с API ключом
        client = Mistral(api_key=self.api_key)
        # Отправка запроса к API для получения ответа от модели
        chat_response = client.chat.complete(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.message,
                },
            ]
        )
        # Получение ответа от модели
        assistant_response = chat_response.choices[0].message.content
        # Испускание сигнала с результатом
        self.result.emit(assistant_response)

# Класс VoiceInputWorker для обработки голосового ввода в отдельном потоке
class VoiceInputWorker(QThread):
    result = pyqtSignal(str)
    error = pyqtSignal(str)

    def run(self):
        # Инициализация распознавателя речи
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        with microphone as source:
            # Настройка уровня шума
            recognizer.adjust_for_ambient_noise(source)
            # Запись аудио
            audio = recognizer.listen(source)

        try:
            # Распознавание речи
            text = recognizer.recognize_google(audio, language="ru-RU")
            # Испускание сигнала с результатом
            self.result.emit(text)
        except sr.UnknownValueError:
            # Испускание сигнала с ошибкой, если речь не распознана
            self.error.emit("Не удалось распознать речь.")
        except sr.RequestError as e:
            # Испускание сигнала с ошибкой, если произошла ошибка сервиса распознавания речи
            self.error.emit(f"Ошибка сервиса распознавания речи; {e}")

# Класс ImageGenerationWorker для обработки генерации изображений в отдельном потоке
class ImageGenerationWorker(QThread):
    result = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, api_url, api_key, secret_key, prompt, width, height):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.secret_key = secret_key
        self.prompt = prompt
        self.width = width
        self.height = height

    def run(self):
        # Инициализация API клиента для генерации изображений
        api = TextToImageAPI(self.api_url, self.api_key, self.secret_key)
        # Получение модели для генерации изображений
        model_id = api.get_model()
        # Генерация изображения
        uuid = api.generate(self.prompt, model_id, self.width, self.height)
        # Проверка статуса генерации изображения
        images = api.check_generation(uuid)
        if images:
            # Испускание сигнала с результатом
            self.result.emit(images[0])
        else:
            # Испускание сигнала с ошибкой
            self.error.emit("Ошибка при генерации изображения.")

# Класс TextToImageAPI для взаимодействия с API генерации изображений
class TextToImageAPI:
    def __init__(self, url, api_key, secret_key):
        self.URL = url
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }

    def get_model(self):
        # Получение списка моделей
        response = requests.get(self.URL + 'key/api/v1/models', headers=self.AUTH_HEADERS)
        data = response.json()
        return data[0]['id']

    def generate(self, prompt, model, width, height):
        # Параметры для генерации изображения
        params = {
            "type": "GENERATE",
            "numImages": 1,
            "width": width,
            "height": height,
            "generateParams": {
                "query": f"{prompt}"
            }
        }

        data = {
            'model_id': (None, model),
            'params': (None, json.dumps(params), 'application/json')
        }
        # Отправка запроса на генерацию изображения
        response = requests.post(self.URL + 'key/api/v1/text2image/run', headers=self.AUTH_HEADERS, files=data)
        data = response.json()
        return data['uuid']

    def check_generation(self, request_id, attempts=10, delay=10):
        # Проверка статуса генерации изображения
        while attempts > 0:
            response = requests.get(self.URL + 'key/api/v1/text2image/status/' + request_id, headers=self.AUTH_HEADERS)
            data = response.json()
            if data['status'] == 'DONE':
                return data['images']

            attempts -= 1
            time.sleep(delay)

    @staticmethod
    def decode_base64(base64_string):
        # Декодирование base64 строки в изображение
        image_data = base64.b64decode(base64_string)
        return image_data

# Класс ImageGenerationWindow для окна генерации изображений
class ImageGenerationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генерация изображений")
        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        layout = QVBoxLayout(self)

        self.prompt_input = QLineEdit(self)
        self.prompt_input.setPlaceholderText("Введите промпт для генерации изображения...")
        self.prompt_input.setStyleSheet("""
            QLineEdit {
                background-color: #2e2e2e;
                border: 1px solid #3e3e3e;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
            }
        """)
        layout.addWidget(self.prompt_input)

        resolution_layout = QHBoxLayout()

        self.width_combo = QComboBox(self)
        self.width_combo.addItems(["144","240","360","480", "512", "720", "1280", "1366", "1440", "1920", "2560"])
        self.width_combo.setCurrentText("1024")
        self.width_combo.setStyleSheet("""
            QComboBox {
                background-color: #2e2e2e;
                border: 1px solid #3e3e3e;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
            }
        """)
        resolution_layout.addWidget(QLabel("Ширина:"))
        resolution_layout.addWidget(self.width_combo)

        self.height_combo = QComboBox(self)
        self.height_combo.addItems(["144","240","360","480", "512", "720", "1280", "1366", "1440", "1920", "2560"])
        self.height_combo.setCurrentText("1024")
        self.height_combo.setStyleSheet("""
            QComboBox {
                background-color: #2e2e2e;
                border: 1px solid #3e3e3e;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
            }
        """)
        resolution_layout.addWidget(QLabel("Высота:"))
        resolution_layout.addWidget(self.height_combo)

        layout.addLayout(resolution_layout)

        self.generate_button = QPushButton("Сгенерировать изображение", self)
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #778D45;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #5e5e5e;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.3);
            }
        """)
        self.generate_button.clicked.connect(self.generate_image)
        layout.addWidget(self.generate_button)

        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #333333;
                border: 2px solid #3e3e3e;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.image_label)

        self.save_button = QPushButton("Сохранить изображение", self)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #778D45;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #5e5e5e;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.3);
            }
        """)
        self.save_button.clicked.connect(self.save_image)
        layout.addWidget(self.save_button)

    def generate_image(self):
        # Получение промпта и разрешения изображения
        prompt = self.prompt_input.text()
        if prompt.strip() == "":
            return

        width = int(self.width_combo.currentText())
        height = int(self.height_combo.currentText())

        # Запуск рабочего потока для генерации изображения
        self.image_generation_worker = ImageGenerationWorker(
            'https://api-key.fusionbrain.ai/',
            'A16907ADAD3C37881EEE799BC1C3FE88',
            'EA49C720BFA29E1AA9219AA45CECF9BC',
            prompt,
            width,
            height
        )
        self.image_generation_worker.result.connect(self.handle_image_generation_result)
        self.image_generation_worker.error.connect(self.handle_image_generation_error)
        self.image_generation_worker.start()

    def handle_image_generation_result(self, base64_string):
        # Декодирование base64 строки в изображение и отображение его
        image_data = TextToImageAPI.decode_base64(base64_string)
        pixmap = QPixmap()
        pixmap.loadFromData(image_data)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def handle_image_generation_error(self, error_message):
        # Обработка ошибки генерации изображения
        QMessageBox.critical(self, "Ошибка", error_message)

    def save_image(self):
        # Сохранение изображения
        pixmap = self.image_label.pixmap()
        if pixmap:
            file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", "", "Images (*.png *.jpg *.bmp)")
            if file_path:
                pixmap.save(file_path)
                QMessageBox.information(self, "Сохранено", f"Изображение успешно сохранено в {file_path}")

# Основное окно приложения
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RevAI")
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        self.db_name = 'chat_history.db'
        if not os.path.exists(self.db_name):
            self.create_database()

        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

        self.api_key = 'yZ7IjIRJ8jyaCvfXFu4NAIfTq3VB0hdH'
        self.model = "pixtral-12b-2409"

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        left_panel_layout = QVBoxLayout()
        self.chat_list = QListWidget()
        self.chat_list.setStyleSheet("""
            QListWidget {
                background-color: #333333;
                border: none;
                padding: 10px;
                border: 2px solid #3e3e3e;
                border-radius: 8px;
            }
            QListWidgetItem {
                color: #778D45;
                border: 2px solid #3e3e3e;
                border-radius: 8px;
                padding: 20px;
                margin: 10px;
                background-color: #778D45;
                border-radius: 8px;
            }
            QListWidget::item:selected {
                background-color: #5a6e57;
                color: #ffffff;
                border: 2px solid #3e3e3e;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
                border-radius: 8px;
            }
        """)
        self.load_chats()
        left_panel_layout.addWidget(self.chat_list)

        self.generate_image_button = QPushButton("🖼️")
        self.generate_image_button.setStyleSheet("""
            QPushButton {
                background-color: #3e3e3e;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #5e5e5e;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.3);
            }
        """)
        self.generate_image_button.setFixedSize(40, 40)
        self.generate_image_button.setToolTip("Генерация изображений")
        self.generate_image_button.clicked.connect(self.open_image_generation_window)

        self.add_chat_button = QPushButton("➕")
        self.add_chat_button.setStyleSheet("""
            QPushButton {
                background-color: #3e3e3e;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #5e5e5e;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.3);
            }
        """)
        self.add_chat_button.setFixedSize(40, 40)
        self.add_chat_button.setToolTip("Добавить чат")
        self.add_chat_button.clicked.connect(self.add_chat)

        self.remove_chat_button = QPushButton("➖")
        self.remove_chat_button.setStyleSheet("""
            QPushButton {
                background-color: #3e3e3e;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #5e5e5e;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.3);
            }
        """)
        self.remove_chat_button.setFixedSize(40, 40)
        self.remove_chat_button.setToolTip("Удалить чат")
        self.remove_chat_button.clicked.connect(self.remove_chat)

        self.clear_chat_button = QPushButton("🗑️")
        self.clear_chat_button.setStyleSheet("""
            QPushButton {
                background-color: #3e3e3e;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #5e5e5e;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.3);
            }
        """)
        self.clear_chat_button.setFixedSize(40, 40)
        self.clear_chat_button.setToolTip("Очистить чат")
        self.clear_chat_button.clicked.connect(self.clear_chat)

        self.voice_input_button = QPushButton("🎤")
        self.voice_input_button.setStyleSheet("""
            QPushButton {
                background-color: #3c7280;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #5e5e5e;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.3);
            }
        """)
        self.voice_input_button.setFixedSize(40, 40)
        self.voice_input_button.setToolTip("Голосовой ввод")
        self.voice_input_button.clicked.connect(self.start_voice_input)

        self.export_chat_button = QPushButton("📤")
        self.export_chat_button.setStyleSheet("""
            QPushButton {
                background-color: #3e3e3e;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #5e5e5e;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.3);
            }
        """)
        self.export_chat_button.setFixedSize(40, 40)
        self.export_chat_button.setToolTip("Экспорт истории чата")
        self.export_chat_button.clicked.connect(self.export_chat_history)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.generate_image_button)
        button_layout.addWidget(self.add_chat_button)
        button_layout.addWidget(self.remove_chat_button)
        button_layout.addWidget(self.clear_chat_button)
        button_layout.addWidget(self.voice_input_button)
        button_layout.addWidget(self.export_chat_button)
        left_panel_layout.addLayout(button_layout)

        layout.addLayout(left_panel_layout, 2)

        chat_area = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                border: 2px solid #3e3e3e;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                color: #ffffff;
            }
        """)
        chat_area.addWidget(self.chat_display, 8)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите свое сообщение...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #2e2e2e;
                border: 1px solid #3e3e3e;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
            }
        """)
        input_layout.addWidget(self.input_field, 8)

        self.send_button = QPushButton("Отправить")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #778D45;
                border: none ;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                color: #ffffff;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #5e5e5e;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.3);
            }
        """)
        input_layout.addWidget(self.send_button, 2)

        chat_area.addLayout(input_layout, 1)
        layout.addLayout(chat_area, 6)

        self.send_button.clicked.connect(self.send_message)
        self.input_field.returnPressed.connect(self.send_message)
        self.chat_list.currentItemChanged.connect(self.load_chat_history)

    def create_database(self):
        # Создание базы данных
        conn = sqlite3.connect(self.db_name)
        conn.close()

    def create_tables(self):
        # Создание таблиц для хранения истории чатов
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                message TEXT NOT NULL,
                role TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats (id)
            )
        """)
        self.conn.commit()

    def load_chats(self):
        # Загрузка списка чатов из базы данных
        self.cursor.execute("SELECT name FROM chats")
        chats = self.cursor.fetchall()
        for chat in chats:
            self.chat_list.addItem(chat[0])

    def load_chat_history(self, current, previous):
        # Загрузка истории чата из базы данных
        if current:
            chat_name = current.text()
            self.cursor.execute("SELECT id FROM chats WHERE name = ?", (chat_name,))
            chat_id = self.cursor.fetchone()[0]
            self.cursor.execute("SELECT message, role FROM messages WHERE chat_id = ?", (chat_id,))
            messages = self.cursor.fetchall()
            self.chat_display.clear()
            for message, role in messages:
                if role == "user":
                    self.chat_display.append(f"<b>Вы:</b> {message}")
                else:
                    self.chat_display.append(f"<b>ИИ:</b> {message}")

    def send_message(self):
        # Обработка отправки сообщения
        message = self.input_field.text().strip()
        if message:
            current_chat = self.chat_list.currentItem()
            if current_chat:
                chat_name = current_chat.text()
                self.cursor.execute("SELECT id FROM chats WHERE name = ?", (chat_name,))
                chat_id = self.cursor.fetchone()[0]
                self.cursor.execute("INSERT INTO messages (chat_id, message, role) VALUES (?, ?, ?)", (chat_id, message, "user"))
                self.conn.commit()
                self.chat_display.append(f"<b style='color:#3273a8;'>Вы:</b> {message}")
                self.input_field.clear()

                # Запуск рабочего потока для выполнения запроса к API Mistral
                self.worker = Worker(self.api_key, self.model, message)
                self.worker.result.connect(self.handle_api_response)
                self.worker.start()

    def handle_api_response(self, response):
        # Обработка ответа от API Mistral
        current_chat = self.chat_list.currentItem()
        if current_chat:
            chat_name = current_chat.text()
            self.cursor.execute("SELECT id FROM chats WHERE name = ?", (chat_name,))
            chat_id = self.cursor.fetchone()[0]
            self.cursor.execute("INSERT INTO messages (chat_id, message, role) VALUES (?, ?, ?)", (chat_id, response, "assistant"))
            self.conn.commit()
            self.chat_display.append(f"<b style='color:#32a85b;'>ИИ:</b> {response}")

    def show_settings(self):
        # Отображение настроек
        QMessageBox.information(self, "Настройки", "Здесь будут настройки приложения.")

    def open_image_generation_window(self):
        # Открытие окна генерации изображений
        self.image_generation_window = ImageGenerationWindow()
        self.image_generation_window.show()

    def add_chat(self):
        # Добавление нового чата
        chat_name, ok = QInputDialog.getText(self, "Добавить чат", "Введите название чата:")
        if ok and chat_name:
            self.cursor.execute("INSERT INTO chats (name) VALUES (?)", (chat_name,))
            self.conn.commit()
            self.chat_list.addItem(chat_name)

    def remove_chat(self):
        # Удаление выбранного чата
        current_chat = self.chat_list.currentItem()
        if current_chat:
            chat_name = current_chat.text()
            self.cursor.execute("SELECT id FROM chats WHERE name = ?", (chat_name,))
            chat_id = self.cursor.fetchone()[0]
            self.cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            self.cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            self.conn.commit()
            self.chat_list.takeItem(self.chat_list.row(current_chat))
            self.chat_display.clear()

    def clear_chat(self):
        # Очистка выбранного чата
        current_chat = self.chat_list.currentItem()
        if current_chat:
            chat_name = current_chat.text()
            self.cursor.execute("SELECT id FROM chats WHERE name = ?", (chat_name,))
            chat_id = self.cursor.fetchone()[0]
            self.cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            self.conn.commit()
            self.chat_display.clear()

    def start_voice_input(self):
        # Запуск голосового ввода
        self.voice_input_worker = VoiceInputWorker()
        self.voice_input_worker.result.connect(self.handle_voice_input_result)
        self.voice_input_worker.error.connect(self.handle_voice_input_error)
        self.voice_input_worker.start()

    def handle_voice_input_result(self, text):
        # Обработка результата голосового ввода
        self.input_field.setText(text)
        self.send_message()

    def handle_voice_input_error(self, error_message):
        # Обработка ошибки голосового ввода
        self.chat_display.append(error_message)

    def export_chat_history(self):
        # Экспорт истории чата в текстовый файл
        current_chat = self.chat_list.currentItem()
        if current_chat:
            chat_name = current_chat.text()
            self.cursor.execute("SELECT id FROM chats WHERE name = ?", (chat_name,))
            chat_id = self.cursor.fetchone()[0]
            self.cursor.execute("SELECT message, role FROM messages WHERE chat_id = ?", (chat_id,))
            messages = self.cursor.fetchall()

            file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить историю чата", f"{chat_name}.txt", "Text Files (*.txt)")
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as file:
                    for message, role in messages:
                        if role == "user":
                            file.write(f"Вы: {message}\n")
                        else:
                            file.write(f"ИИ: {message}\n")
                QMessageBox.information(self, "Экспорт завершен", f"История чата '{chat_name}' успешно экспортирована в {file_path}")

    def closeEvent(self, event):
        # Закрытие базы данных при завершении работы
        self.conn.close()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
