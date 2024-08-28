from flask import Flask, request, jsonify
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
import cv2
import os

# Cargar el modelo CLIP y el procesador
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

app = Flask(__name__)

def validate_image_with_clip(image_url, text):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    similarity = torch.nn.functional.cosine_similarity(image_embeds, text_embeds).item()
    return similarity

def validate_video_with_clip(video_url, text):
    response = requests.get(video_url)
    video_path = os.path.join('/tmp', 'temp_video.mp4')
    with open(video_path, 'wb') as f:
        f.write(response.content)
    
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        return 0
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    similarity = torch.nn.functional.cosine_similarity(image_embeds, text_embeds).item()
    os.remove(video_path)

    return similarity

@app.route('/validate_image', methods=['POST'])
def validate_image():
    data = request.json
    image_url = data.get('image_url')
    text = data.get('text')
    similarity = validate_image_with_clip(image_url, text)
    return jsonify({'similarity': similarity})

@app.route('/validate_video', methods=['POST'])
def validate_video():
    data = request.json
    video_url = data.get('video_url')
    text = data.get('text')
    similarity = validate_video_with_clip(video_url, text)
    return jsonify({'similarity': similarity})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'Model is available'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
