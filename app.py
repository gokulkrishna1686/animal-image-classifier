import os
from flask import Flask, request, render_template, jsonify
import torch
import torchvision
from PIL import Image
import torch.nn as nn

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
model = torchvision.models.efficientnet_b2(weights=weights).to(device)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1408, out_features=90, bias=True)
)

model.load_state_dict(torch.load('model/animals_model_weights.pth', map_location=device))
model.eval()

class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 
               'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 
               'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 
               'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 
               'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 
               'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 
               'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 
               'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 
               'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 
               'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 
               'wolf', 'wombat', 'woodpecker', 'zebra']

transform = weights.transforms()

os.makedirs('templates', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            image = Image.open(file.stream).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.inference_mode():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
                top3_prob, top3_indices = torch.topk(probabilities, 3)
                predictions = []
                for prob, idx in zip(top3_prob, top3_indices):
                    predictions.append({
                        'animal': class_names[idx.item()].title(),
                        'probability': f"{prob.item()*100:.2f}%"
                    })
                
                return jsonify({'predictions': predictions})
    
    return render_template('upload.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)