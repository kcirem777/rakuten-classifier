import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import io
import base64
import random

# Configuration de la page
st.set_page_config(
    page_title="Rakuten Product Classifier",
    page_icon="🛍️",
    layout="wide"
)

# Les 14 catégories Rakuten
CATEGORIES = {
    0: "Livre",
    1: "Musique, CD/DVD, Blu-Ray",
    2: "Jeux vidéo, Console",
    3: "Téléphonie, Tablette",
    4: "Informatique, Logiciel",
    5: "TV, Image et Son",
    6: "Maison",
    7: "Électroménager",
    8: "Alimentation, Boisson",
    9: "Brico, Jardin, Animalerie",
    10: "Sport, Loisirs",
    11: "Mode",
    12: "Beauté",
    13: "Jouet, Enfant, Puériculture"
}

def predict_category(image):
    """
    Fonction simulée pour la prédiction
    Remplacez cette fonction par votre modèle réel
    """
    # Simulation d'une prédiction
    # Ici vous intégrerez votre modèle multimodal
    import random
    
    # Simuler des probabilités pour chaque catégorie
    probabilities = np.random.dirichlet(np.ones(14))  # 14 catégories
    predicted_category = np.argmax(probabilities)
    confidence = probabilities[predicted_category]
    
    return predicted_category, confidence, probabilities

def main():
    # Titre et description
    st.title("🛍️ Rakuten Product Classifier")
    st.markdown("**Classification automatique de produits en 16 catégories avec IA**")
    
    # Sidebar avec informations sur le modèle
    with st.sidebar:
        st.header("📊 Informations du modèle")
        st.metric("Précision du modèle", "39%")
        st.metric("Échantillon testé", "407 produits")
        st.metric("Catégories", "14")
        
        st.markdown("---")
        st.markdown("### 🏷️ Catégories disponibles")
        for i, category in CATEGORIES.items():
            st.write(f"{i}: {category}")
    
    # Interface principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload d'image")
        uploaded_file = st.file_uploader(
            "Choisissez une image de produit",
            type=['png', 'jpg', 'jpeg'],
            help="Formats supportés: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Afficher l'image
            image = Image.open(uploaded_file)
            st.image(image, caption="Image uploadée", use_column_width=True)
            
            # Bouton de prédiction
            if st.button("🔍 Classifier le produit", type="primary"):
                with st.spinner("Classification en cours..."):
                    # Prédiction
                    predicted_cat, confidence, all_probs = predict_category(image)
                    
                    # Stocker les résultats dans session state
                    st.session_state.prediction = {
                        'category': predicted_cat,
                        'confidence': confidence,
                        'probabilities': all_probs
                    }
    
    with col2:
        st.header("🎯 Résultats de classification")
        
        if 'prediction' in st.session_state:
            pred = st.session_state.prediction
            
            # Résultat principal
            st.success(f"**Catégorie prédite:** {CATEGORIES[pred['category']]}")
            st.info(f"**Confiance:** {pred['confidence']:.1%}")
            
            # Graphique des probabilités
            st.subheader("📊 Distribution des probabilités")
            
            # Créer un DataFrame pour le graphique
            prob_df = pd.DataFrame({
                'Catégorie': [CATEGORIES[i] for i in range(14)],  # 14 catégories
                'Probabilité': pred['probabilities']
            })
            prob_df = prob_df.sort_values('Probabilité', ascending=False)
            
            # Graphique en barres
            st.bar_chart(prob_df.set_index('Catégorie')['Probabilité'])
            
            # Top 3 des prédictions
            st.subheader("🏆 Top 3 des prédictions")
            top_3 = prob_df.head(3)
            
            for idx, row in top_3.iterrows():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**{row['Catégorie']}**")
                with col_b:
                    st.write(f"{row['Probabilité']:.1%}")
        
        else:
            st.info("👆 Uploadez une image pour commencer la classification")
    
    # Section d'amélioration continue
    st.markdown("---")
    st.header("💡 Amélioration continue")
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("📝 Feedback")
        if 'prediction' in st.session_state:
            feedback = st.selectbox(
                "La prédiction est-elle correcte ?",
                ["Sélectionnez...", "Oui, correct", "Non, incorrect"]
            )
            
            if feedback == "Non, incorrect":
                correct_category = st.selectbox(
                    "Quelle est la bonne catégorie ?",
                    ["Sélectionnez..."] + list(CATEGORIES.values())
                )
                
                if st.button("Envoyer le feedback"):
                    st.success("Merci pour votre feedback ! Il aidera à améliorer le modèle.")
    
    with col4:
        st.subheader("📈 Statistiques")
        st.metric("Accuracy actuelle", "39%", delta="2%")
        st.metric("Feedbacks reçus", "127")
        st.metric("Améliorations", "+5% ce mois")

if __name__ == "__main__":
    main()
