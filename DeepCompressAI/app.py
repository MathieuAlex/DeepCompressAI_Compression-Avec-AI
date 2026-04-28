from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import torch
import os
import io
import base64
from PIL import Image
import torchvision.transforms as transforms
import time
import numpy as np
from src import utils
from src import models
from src import losses

app = Flask(__name__)
CORS(app)  # Permet les requêtes cross-origin

# Variables globales pour le modèle
model_components = None
device = None


def initialize_model():
    """Initialise le modèle de compression une seule fois au démarrage"""
    global model_components, device

    device = utils.get_device()
    print(f"Utilisation de l'appareil: {device}")

    # Initialiser les composants du modèle
    compression_encoder = models.Encoder().to(device)
    compression_hyper_encoder = models.HyperEncoder().to(device)
    compression_decoder = models.Decoder().to(device)
    compression_hyper_decoder = models.HyperDecoder().to(device)

    model_components = {
        'compression_encoder': compression_encoder,
        'compression_hyper_encoder': compression_hyper_encoder,
        'compression_decoder': compression_decoder,
        'compression_hyper_decoder': compression_hyper_decoder
    }

    # Charger le modèle préentraîné
    model_path = "saved_models/compression_model_2000000.pt"
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_components['compression_encoder'].load_state_dict(checkpoint['compression_encoder_state_dict'])
        model_components['compression_hyper_encoder'].load_state_dict(
            checkpoint['compression_hyper_encoder_state_dict'])
        model_components['compression_decoder'].load_state_dict(checkpoint['compression_decoder_state_dict'])
        model_components['compression_hyper_decoder'].load_state_dict(
            checkpoint['compression_hyper_decoder_state_dict'])
        print(f"Modèle chargé depuis: {model_path}")
        return True
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        return False


def process_image_compression(image_data):
    """Traite la compression d'une image"""
    global model_components, device

    if model_components is None:
        raise Exception("Modèle non initialisé")

    start_time = time.time()

    # Préparer l'image
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    width, height = image.size

    # Redimensionner pour compatibilité (divisible par 16)
    new_width = (width // 16) * 16
    new_height = (height // 16) * 16

    if new_width != width or new_height != height:
        image = image.resize((new_width, new_height))

    # Transformation
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(image).unsqueeze(0).to(device)

    # Compression
    with torch.no_grad():
        # Encodage
        y = model_components['compression_encoder'](x)
        z = model_components['compression_hyper_encoder'](y)

        # Quantification
        z_hat = model_components['compression_hyper_encoder'].integer_quantization(z)
        y_para = model_components['compression_hyper_decoder'](z_hat)

        mu_y = y_para[:, 0:12]
        sigma_y = 0.01 + torch.abs(y_para[:, 12:24])

        y_res = y - mu_y
        y_hat_res = model_components['compression_encoder'].integer_quantization(y_res)
        y_hat = y_hat_res + mu_y

        # Décodage
        x_hat = model_components['compression_decoder'](y_hat)

    # Calculer les métriques
    spatial_px = x.shape[2] * x.shape[3]
    rate_loss_y = losses.rate_loss(y, mu_y, sigma_y)
    rate_loss_z = losses.rate_loss(z)
    bpp_y = rate_loss_y / spatial_px
    bpp_z = rate_loss_z / spatial_px
    bpp_total = bpp_y + bpp_z

    # MSE et autres métriques
    mse = losses.mse_loss(x, x_hat).item()
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    compression_ratio = 24 / bpp_total.item()

    processing_time = (time.time() - start_time) * 1000  # en millisecondes

    # Convertir l'image compressée en PIL
    x_hat_pil = transforms.ToPILImage()(x_hat.squeeze().cpu())

    # Convertir en base64 pour la transmission
    buffer = io.BytesIO()
    x_hat_pil.save(buffer, format='JPEG', quality=95)
    compressed_b64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        'compressed_image_b64': compressed_b64,
        'bpp': float(bpp_total.item()),
        'compression_ratio': float(compression_ratio),
        'mse': float(mse),
        'psnr': float(psnr),
        'processing_time': float(processing_time),
        'original_size': len(image_data),
        'compressed_size': len(buffer.getvalue()),
        'new_dimensions': f"{new_width}x{new_height}"
    }


@app.route('/')
def index():
    """Servir la page principale"""
    try:
        with open('inference.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return "Fichier inference.html non trouvé", 404


@app.route('/compress', methods=['POST'])
def compress_image_endpoint():
    """Endpoint pour compresser une image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400

        # Lire les données de l'image
        image_data = file.read()

        # Traiter la compression
        results = process_image_compression(image_data)

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """Vérifier l'état du serveur"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_components is not None
    })


if __name__ == '__main__':
    print("Initialisation du serveur...")

    # Vérifier que le modèle peut être chargé
    if not os.path.exists("saved_models/compression_model_2000000.pt"):
        rar_path = "saved_models/compression_model_2000000.rar"
        if os.path.exists(rar_path):
            print(f"ATTENTION: Le modèle est au format RAR ({rar_path})")
            print("Veuillez extraire le fichier avant de lancer le serveur.")
            exit(1)
        else:
            print("ERREUR: Fichier modèle non trouvé")
            exit(1)

    # Initialiser le modèle
    if initialize_model():
        print("Modèle initialisé avec succès!")
        print("Serveur démarré sur http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Erreur lors de l'initialisation du modèle")
        exit(1)