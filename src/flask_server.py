import os
from flask import Flask
from flask import request
import tempfile
from inference_file import inference


app = Flask(__name__)

port = int(os.environ.get("PORT", 5000))


@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        f = request.files["audio_file"]
        file_dir = tempfile.gettempdir() + "/file.m4a"
        f.save(file_dir)
        transcription = inference(file_dir)
        print(transcription)
        return transcription


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
