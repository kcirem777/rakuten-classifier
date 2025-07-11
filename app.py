import streamlit as st
import joblib
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# ------------------- Chargement des objets -------------------
# Charger le modèle de classification et le vectorizer TF-IDF
model_path = "logistic_model.pkl"
tfidf_path = "tfidf_vectorizer.pkl"

clf = joblib.load(model_path)
vectorizer = joblib.load(tfidf_path)

# ------------------- Préparation du modèle ResNet50 -------------------
resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()

image_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def image_to_vec(uploaded_image):
    try:
        image = Image.open(uploaded_image).convert('RGB')
        image = image_preprocess(image).unsqueeze(0)
        with torch.no_grad():
            vec = resnet_model(image)
        return vec.squeeze().numpy()
    except:
        return np.zeros(2048)

# ------------------- Interface Streamlit -------------------
st.title("🔍 Prédiction de catégorie produit Rakuten (multimodal)")

# Upload de l'image
uploaded_image = st.file_uploader("📸 Téléverse une image produit", type=["jpg", "png"])

# Saisie du texte
designation = st.text_input("✏️ Désignation produit")
description = st.text_area("📝 Description produit")

# Bouton de prédiction
if st.button("Prédire la catégorie"):
    if uploaded_image is not None and (designation or description):
        # 1. Transformer l'image
        image_vec = image_to_vec(uploaded_image)

        # 2. Transformer le texte
        full_text = (designation + ' ' + description).strip()
        text_vec = vectorizer.transform([full_text]).toarray().squeeze()

        # 3. Fusion vecteurs texte + image
        full_vec = np.concatenate([text_vec, image_vec]).reshape(1, -1)

        # 4. Prédiction
        pred = clf.predict(full_vec)[0]

        st.success(f"Catégorie prédite : **{pred}**")
    else:
        st.warning("Merci de fournir à la fois une image et un minimum de texte.")
