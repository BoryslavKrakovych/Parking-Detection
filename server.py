# === Імпорти необхідних модулів ===
from flask import Flask, request, jsonify, send_file, render_template  # Flask і HTTP-функції
import os                # Робота з файловою системою
import datetime          # Для створення мітки часу в назві файлів
import glob              # Для пошуку файлів за шаблоном

# === Ініціалізація Flask-додатку ===
app = Flask(__name__)

# === Раніше використовувалася JSON-відповідь на GET-запит "/" ===
# Закоментовано, бо нижче використовується HTML-інтерфейс
# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({
#         "status": "Server is running",
#         "endpoints": ["/upload_frame", "/upload_status"]
#     })

# === Прийом кадру з клієнта ===
@app.route('/upload_frame', methods=['POST'])
def receive_frame():
    file = request.files.get('frame')  # Отримуємо файл із поля "frame"
    if file:
        # Створюємо унікальну назву для зображення за поточним часом
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"received/frame_{ts}.jpg"  # Шлях до збереження

        os.makedirs("received", exist_ok=True)  # Створюємо директорію, якщо її нема
        file.save(save_path)                    # Зберігаємо файл

        print(f"[SERVER] Кадр збережено: {save_path}")
        return {"status": "ok"}, 200  # Відповідь OK
    return {"status": "no frame received"}, 400  # Якщо файл не надіслано

# === Прийом JSON-статусу з клієнта ===
@app.route('/upload_status', methods=['POST'])
def receive_status():
    data = request.json  # Отримуємо JSON-об'єкт
    print(f"[SERVER] Отримано статуси: {data}")
    return {"status": "ok"}, 200  # Підтверджуємо прийом

# === Відображення HTML-інтерфейсу з кадром ===
@app.route("/")
def index():
    return render_template("viewer.html")  # Рендеримо HTML зі сторінки templates/viewer.html

# === Віддаємо останній кадр для перегляду в браузері ===
@app.route("/latest")
def latest_frame():
    # Знаходимо всі збережені кадри
    files = sorted(glob.glob("received/frame_*.jpg"))
    if files:
        return send_file(files[-1], mimetype='image/jpeg')  # Віддаємо останній (найновіший)
    return "No frames received", 404  # Якщо нема кадрів — повертаємо 404

# === Запуск Flask-сервера ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # Доступ з усіх IP (у локальній мережі також)
