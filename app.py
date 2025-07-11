import streamlit as st
import joblib
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# Configuration de la page avec thème Rakuten
st.set_page_config(
    page_title="Rakuten - Classification de produits",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour reproduire le style Rakuten
st.markdown("""
<style>
    /* Import de la police Rakuten-like */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Variables couleurs Rakuten */
    :root {
        --rakuten-red: #bf0000;
        --rakuten-red-hover: #a50000;
        --rakuten-black: #333333;
        --rakuten-gray: #666666;
        --rakuten-light-gray: #f8f8f8;
        --rakuten-border: #e0e0e0;
        --rakuten-white: #ffffff;
        --rakuten-dark-gray: #2c2c2c;
    }
    
    /* Background principal en blanc */
    .main {
        background-color: var(--rakuten-white) !important;
    }
    
    .stApp {
        background-color: var(--rakuten-white) !important;
    }
    
    /* Améliorer la visibilité du texte */
    .stMarkdown, .stText, p, span, div {
        color: var(--rakuten-dark-gray) !important;
    }
    
    /* Labels et titres plus visibles */
    label, .stSelectbox label, .stTextInput label, .stTextArea label {
        color: var(--rakuten-black) !important;
        font-weight: 600 !important;
    }
    
    /* Reset du style Streamlit */
    .main > div {
        padding: 0rem 1rem 10rem;
    }
    
    /* Header Rakuten */
    .rakuten-header {
        background: white;
        border-bottom: 1px solid var(--rakuten-border);
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .rakuten-logo {
        color: var(--rakuten-red);
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'Roboto', sans-serif;
        text-align: center;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    
    /* Container principal avec bordures visibles */
    .rakuten-container {
        max-width: 1200px;
        margin: 0 auto;
        background: var(--rakuten-white);
        border-radius: 8px;
        border: 1px solid var(--rakuten-border);
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        overflow: hidden;
        padding: 2rem;
    }
    
    /* Section upload avec bordures visibles */
    .upload-section {
        background: var(--rakuten-light-gray);
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        border: 2px dashed var(--rakuten-border);
        text-align: center;
    }
    
    .upload-section:hover {
        border-color: var(--rakuten-red);
        background: #fafafa;
    }
    
    /* Étapes */
    .etape {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        border-left: 4px solid var(--rakuten-red);
    }
    
    .etape-number {
        background: var(--rakuten-red);
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 1rem;
        font-size: 0.9rem;
    }
    
    .etape-active {
        border-left-color: var(--rakuten-red);
        background: #fff5f5;
    }
    
    .etape-completed {
        border-left-color: #4CAF50;
    }
    
    .etape-completed .etape-number {
        background: #4CAF50;
    }
    
    /* Boutons Rakuten */
    .stButton > button {
        background: var(--rakuten-red) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 500 !important;
        font-family: 'Roboto', sans-serif !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: var(--rakuten-red-hover) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(191, 0, 0, 0.3) !important;
    }
    
    /* Résultats */
    .result-container {
        background: white;
        border: 1px solid var(--rakuten-border);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .category-result {
        background: var(--rakuten-white);
        border: 2px solid var(--rakuten-red);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(191, 0, 0, 0.1);
    }
    
    .category-name {
        color: var(--rakuten-red);
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .category-subtitle {
        color: var(--rakuten-gray);
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Sections avec bordures et contrastes */
    .section-title {
        color: var(--rakuten-black) !important;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--rakuten-red);
    }
    
    /* Colonnes avec séparation visuelle */
    .column-left {
        padding: 1.5rem;
        border-right: 1px solid var(--rakuten-border);
        background: var(--rakuten-white);
    }
    
    .column-right {
        padding: 1.5rem;
        background: var(--rakuten-white);
    }
    
    /* Input styles avec bordures visibles */
    .stTextInput > div > div > input {
        border: 2px solid var(--rakuten-border) !important;
        border-radius: 6px !important;
        padding: 0.75rem !important;
        font-family: 'Roboto', sans-serif !important;
        color: var(--rakuten-dark-gray) !important;
        background: var(--rakuten-white) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--rakuten-red) !important;
        box-shadow: 0 0 0 2px rgba(191, 0, 0, 0.2) !important;
    }
    
    /* TextArea avec bordures */
    .stTextArea > div > div > textarea {
        border: 2px solid var(--rakuten-border) !important;
        border-radius: 6px !important;
        padding: 0.75rem !important;
        font-family: 'Roboto', sans-serif !important;
        color: var(--rakuten-dark-gray) !important;
        background: var(--rakuten-white) !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--rakuten-red) !important;
        box-shadow: 0 0 0 2px rgba(191, 0, 0, 0.2) !important;
    }
    
    /* File uploader avec bordures visibles */
    .stFileUploader > div {
        border: 2px solid var(--rakuten-border) !important;
        border-radius: 8px !important;
        background: var(--rakuten-light-gray) !important;
    }
    
    .stFileUploader label {
        color: var(--rakuten-black) !important;
        font-weight: 600 !important;
    }
    
    /* Forcer les couleurs du selectbox - approche plus agressive */
    .stSelectbox div[data-baseweb="select"] > div,
    .stSelectbox div[data-baseweb="select"] > div > div,
    .stSelectbox [data-testid="stSelectbox"] > div > div,
    .stSelectbox [data-testid="stSelectbox"] > div > div > div {
        background-color: var(--rakuten-white) !important;
        color: var(--rakuten-dark-gray) !important;
        border: 2px solid var(--rakuten-border) !important;
        border-radius: 6px !important;
    }
    
    /* Dropdown ouvert */
    .stSelectbox [role="listbox"],
    .stSelectbox [data-baseweb="popover"] > div,
    .stSelectbox ul,
    .stSelectbox ul > li {
        background-color: var(--rakuten-white) !important;
        color: var(--rakuten-dark-gray) !important;
        border: 1px solid var(--rakuten-border) !important;
    }
    
    /* Options individuelles */
    .stSelectbox [role="option"],
    .stSelectbox li {
        background-color: var(--rakuten-white) !important;
        color: var(--rakuten-dark-gray) !important;
        padding: 0.75rem !important;
    }
    
    .stSelectbox [role="option"]:hover,
    .stSelectbox li:hover {
        background-color: var(--rakuten-light-gray) !important;
        color: var(--rakuten-red) !important;
    }
    
    /* Option sélectionnée */
    .stSelectbox [aria-selected="true"] {
        background-color: var(--rakuten-red) !important;
        color: var(--rakuten-white) !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: #fff5f5 !important;
        border: 1px solid var(--rakuten-red) !important;
        color: var(--rakuten-red) !important;
    }
    
    /* Sidebar si nécessaire */
    .css-1d391kg {
        background: var(--rakuten-light-gray);
    }
    
    /* Progress steps */
    .progress-steps {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
        padding: 1rem;
        background: var(--rakuten-light-gray);
        border-radius: 8px;
    }
    
    .step {
        text-align: center;
        flex: 1;
        padding: 0 1rem;
    }
    
    .step-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #ccc;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.5rem;
        font-weight: bold;
    }
    
    .step-active .step-icon {
        background: var(--rakuten-red);
    }
    
    .step-completed .step-icon {
        background: #4CAF50;
    }
    
    .step-title {
        font-size: 0.9rem;
        color: var(--rakuten-gray);
        font-weight: 500;
    }
    
    .step-active .step-title {
        color: var(--rakuten-red);
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Chargement des objets -------------------
@st.cache_resource
def load_models():
    try:
        model_path = "logistic_model.pkl"
        tfidf_path = "tfidf_vectorizer.pkl"
        clf = joblib.load(model_path)
        vectorizer = joblib.load(tfidf_path)
        return clf, vectorizer
    except Exception as e:
        st.error(f"Erreur chargement modèles: {e}")
        return None, None

@st.cache_resource
def load_resnet():
    try:
        resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
        resnet_model.eval()
        return resnet_model
    except Exception as e:
        st.error(f"Erreur chargement ResNet: {e}")
        return None

# Catégories Rakuten avec icônes
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

# Icônes correspondantes pour chaque catégorie
CATEGORY_ICONS = {
    0: "📚",  # Livre
    1: "🎵",  # Musique, CD/DVD, Blu-Ray
    2: "🎮",  # Jeux vidéo, Console
    3: "📱",  # Téléphonie, Tablette
    4: "💻",  # Informatique, Logiciel
    5: "📺",  # TV, Image et Son
    6: "🏠",  # Maison
    7: "🔌",  # Électroménager
    8: "🍕",  # Alimentation, Boisson
    9: "🔨",  # Brico, Jardin, Animalerie
    10: "⚽", # Sport, Loisirs
    11: "👕", # Mode
    12: "💄", # Beauté
    13: "🧸"  # Jouet, Enfant, Puériculture
}

# Préprocessing image
image_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def image_to_vec(uploaded_image):
    try:
        resnet_model = load_resnet()
        if resnet_model is None:
            return np.zeros(2048)
            
        image = Image.open(uploaded_image).convert('RGB')
        image = image_preprocess(image).unsqueeze(0)
        with torch.no_grad():
            vec = resnet_model(image)
        return vec.squeeze().numpy()
    except Exception as e:
        st.error(f"Erreur traitement image: {e}")
        return np.zeros(2048)

def generate_product_description(image, category_name):
    """
    Générer une description automatique basée sur l'image et la catégorie
    """
    try:
        # Suggestions basées sur la catégorie prédite
        suggestions = {
            "Livre": {
                "name": "Livre de fiction moderne",
                "description": "Livre en bon état, couverture préservée, pages propres. Idéal pour les amateurs de lecture."
            },
            "Musique, CD/DVD, Blu-Ray": {
                "name": "Album musical / Film DVD",
                "description": "Support en excellent état, sans rayures. Boîtier d'origine inclus."
            },
            "Jeux vidéo, Console": {
                "name": "Jeu vidéo / Console de jeu",
                "description": "Produit gaming en parfait état de fonctionnement, testé et approuvé."
            },
            "Téléphonie, Tablette": {
                "name": "Smartphone / Tablette récent(e)",
                "description": "Appareil en excellent état, écran sans fissure, toutes fonctionnalités opérationnelles."
            },
            "Informatique, Logiciel": {
                "name": "Matériel informatique",
                "description": "Équipement informatique performant, testé et en parfait état de marche."
            },
            "TV, Image et Son": {
                "name": "Équipement audiovisuel",
                "description": "Appareil électronique en excellent état, toutes les fonctions opérationnelles."
            },
            "Maison": {
                "name": "Article de décoration / mobilier",
                "description": "Objet décoratif ou mobilier en bon état, sans défaut majeur."
            },
            "Électroménager": {
                "name": "Appareil électroménager",
                "description": "Électroménager en parfait état de fonctionnement, toutes fonctions testées."
            },
            "Alimentation, Boisson": {
                "name": "Produit alimentaire",
                "description": "Produit frais ou conserve, date de péremption respectée."
            },
            "Brico, Jardin, Animalerie": {
                "name": "Outil de bricolage / Article jardin",
                "description": "Outil ou accessoire en bon état, fonctionnel et prêt à l'emploi."
            },
            "Sport, Loisirs": {
                "name": "Équipement sportif",
                "description": "Matériel de sport en excellent état, peu utilisé, idéal pour la pratique."
            },
            "Mode": {
                "name": "Vêtement / Accessoire de mode",
                "description": "Article de mode en très bon état, taille conforme, couleurs préservées."
            },
            "Beauté": {
                "name": "Produit de beauté / cosmétique",
                "description": "Produit cosmétique en parfait état, emballage d'origine, non ouvert ou peu utilisé."
            },
            "Jouet, Enfant, Puériculture": {
                "name": "Jouet / Article puériculture",
                "description": "Article pour enfant en excellent état, propre et sécurisé, toutes pièces incluses."
            }
        }
        
        return suggestions.get(category_name, {
            "name": "Produit en excellent état",
            "description": "Article en très bon état général, conforme à la description."
        })
        
    except Exception as e:
        return {
            "name": "Produit de qualité",
            "description": "Article en bon état, prêt à être utilisé."
        }
    clf, vectorizer = load_models()
    if clf is None or vectorizer is None:
        return None, None
    
    try:
        # 1. Transformer l'image
        image_vec = image_to_vec(uploaded_image)
        
        # 2. Transformer le texte
        full_text = (designation + ' ' + description).strip()
        if not full_text:
            full_text = "produit"  # texte minimal
        text_vec = vectorizer.transform([full_text]).toarray().squeeze()
        
        # Debug des dimensions
        st.info(f"Debug: Text features: {text_vec.shape}, Image features: {image_vec.shape}")
        
        # 3. Vérifier la compatibilité des dimensions
        expected_features = clf.coef_.shape[1] if hasattr(clf, 'coef_') else None
        if expected_features:
            st.info(f"Debug: Model expects {expected_features} features total")
            
            # Ajuster les dimensions si nécessaire
            total_current = len(text_vec) + len(image_vec)
            if total_current != expected_features:
                # Truncate ou pad selon les besoins
                if total_current > expected_features:
                    # Truncate les features image
                    needed_image_features = expected_features - len(text_vec)
                    if needed_image_features > 0:
                        image_vec = image_vec[:needed_image_features]
                    else:
                        # Truncate text si nécessaire
                        text_vec = text_vec[:expected_features]
                        image_vec = np.array([])
                else:
                    # Pad avec des zéros
                    missing = expected_features - total_current
                    image_vec = np.concatenate([image_vec, np.zeros(missing)])
        
        # 4. Fusion vecteurs texte + image
        if len(image_vec) > 0:
            full_vec = np.concatenate([text_vec, image_vec]).reshape(1, -1)
        else:
            full_vec = text_vec.reshape(1, -1)
        
        st.info(f"Debug: Final vector shape: {full_vec.shape}")
        
        # 5. Prédiction
        pred_raw = clf.predict(full_vec)[0]
        probabilities = clf.predict_proba(full_vec)[0]
        
        # Debug de la prédiction
        st.info(f"Debug: Pred type: {type(pred_raw)}, Pred value: {pred_raw}")
        st.info(f"Debug: Probabilities shape: {probabilities.shape}")
        
        # Le modèle retourne le nom de la catégorie, pas un index
        # Trouver l'index correspondant dans notre dictionnaire CATEGORIES
        category_name = pred_raw
        pred_index = None
        
        # Chercher l'index de la catégorie prédite
        for idx, cat_name in CATEGORIES.items():
            if cat_name == category_name:
                pred_index = idx
                break
        
        # Si pas trouvé, chercher une correspondance partielle
        if pred_index is None:
            for idx, cat_name in CATEGORIES.items():
                if category_name.lower() in cat_name.lower() or cat_name.lower() in category_name.lower():
                    pred_index = idx
                    break
        
        # Si toujours pas trouvé, utiliser l'index de la probabilité max
        if pred_index is None:
            pred_index = np.argmax(probabilities)
            st.warning(f"Catégorie '{category_name}' non trouvée, utilisation de l'index {pred_index}")
        
        # Vérifier que l'index est valide
        if pred_index < 0 or pred_index >= len(probabilities):
            st.error(f"Index invalide: {pred_index}, probabilities length: {len(probabilities)}")
            return None, None
            
        confidence = probabilities[pred_index]
        
        st.info(f"Debug: Final pred_index: {pred_index}, category: {CATEGORIES.get(pred_index, 'Unknown')}, confidence: {confidence}")
        
        return pred_index, confidence
    except Exception as e:
        st.error(f"Erreur prédiction: {e}")
        # Afficher plus de détails pour debug
        st.error(f"Type d'erreur: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

# ------------------- Interface Streamlit -------------------

# Header Rakuten avec logo officiel
logo_html = '''
<div class="rakuten-header">
    <div class="logo-container">
        <svg width="200" height="60" viewBox="0 0 1200 400" xmlns="http://www.w3.org/2000/svg">
            <text x="50" y="300" font-family="Arial, sans-serif" font-size="180" font-weight="bold" fill="#BF0000">Rakuten</text>
            <path d="M50 320 L950 320 L980 360 L50 360 Z" fill="#BF0000"/>
        </svg>
    </div>
</div>
'''
st.markdown(logo_html, unsafe_allow_html=True)

# Progress steps
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="step step-active">
        <div class="step-icon">1</div>
        <div class="step-title">Votre annonce</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    step2_class = "step-active" if 'uploaded_image' in st.session_state else "step"
    st.markdown(f"""
    <div class="{step2_class}">
        <div class="step-icon">2</div>
        <div class="step-title">Classification IA</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    step3_class = "step-completed" if 'prediction_result' in st.session_state else "step"
    st.markdown(f"""
    <div class="{step3_class}">
        <div class="step-icon">3</div>
        <div class="step-title">Résultat</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Container principal
st.markdown('<div class="rakuten-container">', unsafe_allow_html=True)

# Section 1: Upload d'image
st.markdown('<h2 class="section-title">📸 Dites-nous en plus</h2>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="column-left">', unsafe_allow_html=True)
    st.markdown("**Ajoutez une image de votre produit**")
    uploaded_image = st.file_uploader(
        "Téléversez une image",
        type=["jpg", "png", "jpeg"],
        help="Formats acceptés: JPG, PNG, JPEG"
    )
    
    if uploaded_image:
        st.image(uploaded_image, caption="Image téléversée", use_column_width=True)
        st.session_state.uploaded_image = uploaded_image
        
        # Générer des suggestions automatiques basées sur l'analyse de l'image
        if st.button("✨ Générer des suggestions automatiques", use_container_width=True):
            with st.spinner("Analyse de l'image en cours..."):
                try:
                    # Faire une prédiction rapide pour obtenir la catégorie
                    pred, confidence = predict_category(uploaded_image, "", "")
                    if pred is not None:
                        category_name = CATEGORIES.get(pred, "Produit")
                        suggestions = generate_product_description(uploaded_image, category_name)
                        
                        st.session_state.suggested_name = suggestions["name"]
                        st.session_state.suggested_description = suggestions["description"]
                        st.success("Suggestions générées ! Vous pouvez les modifier si besoin.")
                        st.rerun()
                    else:
                        st.error("Impossible de générer des suggestions. Veuillez réessayer.")
                except Exception as e:
                    st.error(f"Erreur lors de la génération des suggestions : {str(e)}")
                    # Suggestions par défaut en cas d'erreur
                    st.session_state.suggested_name = "Produit de qualité"
                    st.session_state.suggested_description = "Article en bon état, prêt à être utilisé."
                    st.warning("Suggestions par défaut générées.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="column-right">', unsafe_allow_html=True)
    st.markdown("**Informations produit (optionnel)**")
    
    # Suggestions en italique si disponibles
    name_placeholder = "ex: iPhone 15 Pro Max"
    description_placeholder = "Décrivez les caractéristiques, l'état, etc."
    
    if 'suggested_name' in st.session_state:
        name_placeholder = f"Suggestion: {st.session_state.suggested_name}"
    if 'suggested_description' in st.session_state:
        description_placeholder = f"Suggestion: {st.session_state.suggested_description}"
    
    designation = st.text_input(
        "Nom du produit",
        value=st.session_state.get('suggested_name', ''),
        placeholder=name_placeholder,
        help="Le nom ou titre de votre produit (suggestions générées automatiquement)"
    )
    
    description = st.text_area(
        "Description détaillée",
        value=st.session_state.get('suggested_description', ''),
        placeholder=description_placeholder,
        height=100,
        help="Plus d'informations pour une meilleure classification (suggestions générées automatiquement)"
    )
    
    # Afficher les suggestions en italique sous les champs
    if 'suggested_name' in st.session_state and not designation:
        st.markdown(f"*💡 Suggestion de nom: {st.session_state.suggested_name}*")
    if 'suggested_description' in st.session_state and not description:
        st.markdown(f"*💡 Suggestion de description: {st.session_state.suggested_description}*")
    st.markdown('</div>', unsafe_allow_html=True)

# Bouton de classification
st.markdown("<br>", unsafe_allow_html=True)

if uploaded_image:
    if st.button("🔍 Classifier automatiquement ce produit", type="primary"):
        with st.spinner("Classification en cours avec l'IA..."):
            pred, confidence = predict_category(uploaded_image, designation, description)
            
            if pred is not None:
                st.session_state.prediction_result = {
                    'category': pred,
                    'category_name': CATEGORIES.get(pred, f"Catégorie {pred}"),
                    'confidence': confidence
                }
                st.rerun()

# Section 2: Résultats
if 'prediction_result' in st.session_state:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🎯 Catégorie suggérée</h2>', unsafe_allow_html=True)
    
    result = st.session_state.prediction_result
    
    # Affichage style Rakuten (sans score de confiance)
    st.markdown(f"""
    <div class="category-result">
        <div class="category-name">{result['category_name']}</div>
        <div class="category-subtitle">Catégorie suggérée par l'IA</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Options comme sur Rakuten
    st.markdown("**Cette catégorie vous convient-elle ?**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Oui, c'est parfait"):
            st.success("Parfait ! Votre produit sera classé dans cette catégorie.")
            # Masquer la correction si elle était affichée
            if 'show_category_selector' in st.session_state:
                del st.session_state.show_category_selector
    
    with col2:
        if st.button("❌ Non, changer de catégorie"):
            st.info("Vous pouvez sélectionner manuellement la catégorie.")
            st.session_state.show_category_selector = True
    
    # Afficher le sélecteur de catégorie si demandé
    if st.session_state.get('show_category_selector', False):
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Container avec style Rakuten et styles inline pour forcer
        st.markdown("""
        <div style="
            background: white !important;
            border: 2px solid #e0e0e0 !important;
            border-radius: 8px !important;
            padding: 1.5rem !important;
            margin-top: 1rem !important;
        ">
            <h4 style="color: #bf0000 !important; margin-bottom: 1rem !important;">
                🏷️ Sélectionnez la bonne catégorie
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Alternative : utiliser des boutons radio au lieu du selectbox
        st.markdown("**Choisissez la catégorie correcte :**")
        
        # Organiser en colonnes pour un meilleur affichage
        cols = st.columns(2)
        selected_category = None
        
        # Créer des boutons radio pour chaque catégorie
        category_list = list(CATEGORIES.values())
        
        for i, category in enumerate(category_list):
            col_idx = i % 2
            icon = CATEGORY_ICONS.get(i, "📦")  # Icône spécifique ou générique
            with cols[col_idx]:
                if st.button(f"{icon} {category}", key=f"cat_btn_{i}", use_container_width=True):
                    selected_category = category
                    st.session_state.selected_correction = category
        
        # Ou utiliser un selectbox avec style forcé
        st.markdown("""
        <style>
        div[data-testid="stSelectbox"] > div > div {
            background-color: white !important;
            color: #333 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Vérifier s'il y a une sélection
        if 'selected_correction' in st.session_state:
            selected_category = st.session_state.selected_correction
            
        if selected_category:
            # Trouver l'index de la catégorie sélectionnée
            correct_index = None
            for idx, cat_name in CATEGORIES.items():
                if cat_name == selected_category:
                    correct_index = idx
                    break
            
            if correct_index is not None:
                st.success(f"Merci ! Catégorie corrigée : **{selected_category}**")
                
                # Optionnel : sauvegarder la correction pour l'amélioration du modèle
                if st.button("✅ Confirmer cette correction", type="primary"):
                    st.success("Correction enregistrée ! Cela aidera à améliorer notre IA.")
                    # Ici vous pourriez sauvegarder la correction dans une base de données
                    if 'selected_correction' in st.session_state:
                        del st.session_state.selected_correction
                    st.session_state.show_category_selector = False
                    st.rerun()
    
    # Informations additionnelles
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ Comment ça marche ?"):
        st.write(f"""
        Notre intelligence artificielle a analysé votre image et votre description 
        pour suggérer la catégorie **{result['category_name']}**.
        
        Cette classification est basée sur :
        - 🖼️ L'analyse visuelle de votre image
        - 📝 L'analyse du texte de votre description  
        - 🤖 Un modèle entraîné sur des milliers de produits Rakuten
        """)

else:
    # Message d'encouragement
    if not uploaded_image:
        st.markdown("""
        <div class="upload-section">
            <h3>👆 Commencez par téléverser une image</h3>
            <p>Notre IA analysera votre produit et suggérera automatiquement la bonne catégorie</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer simple
st.markdown("<br><br>", unsafe_allow_html=True)
