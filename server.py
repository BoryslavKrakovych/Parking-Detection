from flask import Flask, request, jsonify, send_file, render_template
import os
import datetime
import glob

app = Flask(__name__)

# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({
#         "status": "Server is running",
#         "endpoints": ["/upload_frame", "/upload_status"]
#     })

@app.route('/upload_frame', methods=['POST'])
def receive_frame():
    file = request.files.get('frame')
    if file:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"received/frame_{ts}.jpg"
        os.makedirs("received", exist_ok=True)
        file.save(save_path)
        print(f"[SERVER] Кадр збережено: {save_path}")
        return {"status": "ok"}, 200
    return {"status": "no frame received"}, 400

@app.route('/upload_status', methods=['POST'])
def receive_status():
    data = request.json
    print(f"[SERVER] Отримано статуси: {data}")
    return {"status": "ok"}, 200

@app.route("/")
def index():
    return render_template("viewer.html")

@app.route("/latest")
def latest_frame():
    files = sorted(glob.glob("received/frame_*.jpg"))
    if files:
        return send_file(files[-1], mimetype='image/jpeg')
    return "No frames received", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
