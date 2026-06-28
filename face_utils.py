import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import pickle
from PIL import Image

# Initialize Models globally so they aren't reloaded constantly
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
to_tensor = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])

MODEL_PATH = './face_attendance_model.pkl'

def load_model():
    """Loads the trained model if it exists."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def train_system(dataset_path='./labels'):
    """Extracts embeddings, saves CSVs, and trains the MLP Classifier."""
    mtcnn_train = MTCNN(keep_all=False, device=device) # Single face for training
    
    # Phase 1: Extract and Save CSVs
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            person_embeddings = []
            csv_path = os.path.join(person_dir, f"{person_name}_embeddings.csv")
            
            if not os.path.exists(csv_path):
                for image_name in os.listdir(person_dir):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            img = Image.open(os.path.join(person_dir, image_name)).convert('RGB')
                            face = mtcnn_train(img)
                            if face is None:
                                face = to_tensor(img).unsqueeze(0).to(device)
                                face = (face - 0.5) * 2
                            else:
                                face = face.unsqueeze(0) if face.ndim == 3 else face[0].unsqueeze(0)
                                face = face.to(device)

                            embedding = resnet(face).detach().cpu().numpy()[0]
                            person_embeddings.append(embedding)
                        except Exception:
                            pass
                
                if person_embeddings:
                    pd.DataFrame(person_embeddings).to_csv(csv_path, index=False)

    # Phase 2: Train Model
    X_real, y_real, target_names = [], [], []
    name_to_label = {}
    current_label_id = 0

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            csv_path = os.path.join(person_dir, f"{person_name}_embeddings.csv")
            if os.path.exists(csv_path):
                if person_name not in name_to_label:
                    name_to_label[person_name] = current_label_id
                    target_names.append(person_name)
                    current_label_id += 1
                
                df = pd.read_csv(csv_path)
                embeddings = df.values.tolist()
                X_real.extend(embeddings)
                y_real.extend([name_to_label[person_name]] * len(embeddings))

    if not X_real:
        return False, "No training data found."

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_real), np.array(y_real), test_size=0.20, random_state=42)
    clf = MLPClassifier(hidden_layer_sizes=(512,), max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    data_to_save = {'classifier': clf, 'target_names': target_names, 'name_to_label': name_to_label}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(data_to_save, f)

    return True, f"Successfully trained on {len(target_names)} students."

def predict_face(pil_img, clf, target_names, confidence_threshold=0.70):
    """Predicts faces in a single PIL image."""
    boxes, probs = mtcnn.detect(pil_img)
    results = []
    
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob > 0.90:
                x1, y1, x2, y2 = [int(b) for b in box]
                x1, y1 = max(0, x1), max(0, y1)
                
                try:
                    face_crop = pil_img.crop((x1, y1, x2, y2))
                    face_tensor = to_tensor(face_crop).unsqueeze(0).to(device)
                    face_tensor = (face_tensor - 0.5) * 2
                    
                    embedding = resnet(face_tensor).detach().cpu().numpy()
                    probabilities = clf.predict_proba(embedding)[0]
                    max_prob_index = np.argmax(probabilities)
                    max_prob = probabilities[max_prob_index]
                    
                    if max_prob > confidence_threshold:
                        results.append({'name': target_names[max_prob_index], 'box': (x1, y1, x2, y2), 'prob': max_prob})
                except:
                    pass
    return results
