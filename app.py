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
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
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
    
    .main {
        background-color: var(--rakuten-white) !important;
    }
    
    .stApp {
        background-color: var(--rakuten-white) !important;
    }
    
    .main > div {
        padding: 0rem 1rem 10rem;
    }
    
    .rakuten-header {
        background: white;
        border-bottom: 1px solid var(--rakuten-border);
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    
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
    
    .section-title {
        color: var(--rakuten-black) !important;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--rakuten-red);
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
    
    .step-
