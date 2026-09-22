from model_using import load_model, translate
import config
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

print("Model is loading, it might take a while...")
model, tokenizer, device = load_model(config.MODEL_ID)
print(f"Model is ready to use, using {device}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=['POST'])
def handle_translate():
    data = request.get_json(silent=True) or {}
    pure_text = (data.get("text") or "").strip()
    if not pure_text:
        return jsonify({"translation": ""})
    result = translate(pure_text, model, tokenizer, device)
    return jsonify({"translation": result})



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)