"""
Handwritten Digit Recognizer
Run: python3 app.py  →  http://localhost:7860
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import base64
import json
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = load_model("my_first_ai_model.keras")
print("Model ready — 99.08% accuracy")

app = FastAPI()

# ── Serve the HTML page ───────────────────────────────────────────────────────
with open("digit_recognizer_ui.html", "r") as f:
    HTML = f.read()

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML

# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.post("/run/predict")
async def predict(req: Request):
    body = await req.json()
    image_b64 = body.get("data", [""])[0]

    if not image_b64 or len(image_b64) < 50:
        return JSONResponse({"data": [json.dumps([0.1]*10)]})

    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    img = Image.open(BytesIO(base64.b64decode(image_b64))).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    probs = model.predict(arr, verbose=0)[0].tolist()

    return JSONResponse({"data": [json.dumps(probs)]})

# ── Chat endpoint (no API key needed — answers from built-in knowledge) ───────
@app.post("/ask")
async def ask(req: Request):
    body = await req.json()
    question = body.get("question", "")
    context = body.get("context", "No prediction yet.")

    # Built-in smart answers — no API key required
    q = question.lower()
    if "softmax" in q:
        answer = "Softmax turns the model's 10 raw scores into probabilities that add up to 100%. The highest one is the predicted digit. For example, digit 3 might get 94% while digit 8 gets 5%."
    elif "dropout" in q:
        answer = "Dropout randomly switches off 30% of neurons during training. This forces the network to not rely on any single neuron, so it learns more robust patterns and performs better on new images."
    elif "conv" in q or "conv2d" in q:
        answer = "Conv2D slides a small 3×3 filter across the image, looking for patterns like edges, curves, and corners. The first layer finds simple edges; the second layer combines those into more complex strokes."
    elif "confus" in q or "hard" in q or "pair" in q:
        answer = "The hardest pairs are 1 vs 7 (similar vertical stroke), 3 vs 8 (similar curves), and 4 vs 9 (similar top shape). Bad handwriting or unusual styles cause most of the 92 errors out of 10,000."
    elif "why" in q and ("chose" in q or "choose" in q or "digit" in q or "pick" in q):
        if "No prediction" in context:
            answer = "Draw a digit first and hit Predict, then I can explain why the model chose that result!"
        else:
            answer = f"The model analyzed the shapes and strokes in your drawing against patterns it learned from 60,000 training images. {context} The winning digit had the closest match to its learned filters."
    elif "layer" in q or "architecture" in q or "cnn" in q:
        answer = "The CNN has 8 layers: two Conv2D layers that detect shapes, two MaxPooling layers that shrink the image, a Flatten layer, a Dense(128) layer for classification, Dropout(30%) to prevent overfitting, and a final Dense(10) softmax output."
    elif "mnist" in q or "dataset" in q or "train" in q:
        answer = "MNIST has 70,000 handwritten digit images — 60,000 for training and 10,000 for testing. Each image is 28×28 pixels in grayscale, normalized from 0–255 to 0–1 before being fed to the model."
    elif "accuracy" in q or "99" in q:
        answer = "The model achieved 99.08% test accuracy, correctly classifying 9,908 out of 10,000 images it had never seen before. It trained in about 40 seconds over 5 epochs on a standard laptop."
    elif "relu" in q:
        answer = "ReLU stands for Rectified Linear Unit. It replaces any negative value with zero. This lets the network learn non-linear patterns — without it, stacking layers would be mathematically equivalent to just one layer."
    elif "pooling" in q or "maxpool" in q:
        answer = "MaxPooling takes a 2×2 block of pixels and keeps only the highest value. This shrinks the image by half while keeping the strongest features, making the model faster and less sensitive to small shifts in the drawing."
    else:
        answer = f"Great question! This CNN was trained on 60,000 handwritten digits and achieves 99.08% accuracy with 225,034 parameters. {context} Feel free to ask about any specific layer, concept, or result!"

    return JSONResponse({"answer": answer})

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
port = int(os.environ.get("PORT", 7860))
uvicorn.run(app, host="0.0.0.0", port=port)