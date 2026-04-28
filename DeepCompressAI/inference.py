import torch
import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from src import utils
from src import models
from src import losses


def compress_image(image_path, output_dir="compressed_output", model_path="saved_models/compression_model_2000000.pt"):
    """
    Compresse une image en utilisant un réseau de neurones préentraîné.

    Args:
        image_path (str): Chemin vers l'image à compresser
        output_dir (str): Répertoire où sauvegarder les images compressées
        model_path (str): Chemin vers le modèle préentraîné

    Returns:
        tuple: (image_compressée, bpp, taux_de_compression)
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Déterminer le meilleur appareil pour exécuter le traitement
    device = utils.get_device()
    print(f"Utilisation de l'appareil: {device}")

    # Initialiser les composants du modèle
    compression_encoder = models.Encoder().to(device)
    compression_hyper_encoder = models.HyperEncoder().to(device)
    compression_decoder = models.Decoder().to(device)
    compression_hyper_decoder = models.HyperDecoder().to(device)

    # Dictionnaire pour stocker les composants du modèle
    model_components = {
        'compression_encoder': compression_encoder,
        'compression_hyper_encoder': compression_hyper_encoder,
        'compression_decoder': compression_decoder,
        'compression_hyper_decoder': compression_hyper_decoder
    }

    # Charger le modèle préentraîné
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_components['compression_encoder'].load_state_dict(checkpoint['compression_encoder_state_dict'])
        model_components['compression_hyper_encoder'].load_state_dict(
            checkpoint['compression_hyper_encoder_state_dict'])
        model_components['compression_decoder'].load_state_dict(checkpoint['compression_decoder_state_dict'])
        model_components['compression_hyper_decoder'].load_state_dict(
            checkpoint['compression_hyper_decoder_state_dict'])
        print(f"\nModèle chargé depuis: {model_path}\n")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle depuis {model_path}: {str(e)}")
        return None, None, None

    # Charger et préparer l'image avec redimensionnement pour assurer la compatibilité
    image = Image.open(image_path).convert('RGB')

    # Obtenir les dimensions d'origine
    width, height = image.size

    # Redimensionner à des dimensions divisibles par 16 (pour compatibilité avec le modèle)
    new_width = (width // 16) * 16
    new_height = (height // 16) * 16

    if new_width != width or new_height != height:
        print(f"Redimensionnement de l'image de {width}x{height} à {new_width}x{new_height} pour compatibilité")
        image = image.resize((new_width, new_height))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    x = transform(image).unsqueeze(0).to(device)  # Ajouter une dimension de lot et déplacer vers le device

    # Faire passer l'image dans le modèle
    with torch.no_grad():
        # Encodage
        y = compression_encoder(x)
        z = compression_hyper_encoder(y)

        # Quantification
        z_hat = compression_hyper_encoder.integer_quantization(z)
        y_para = compression_hyper_decoder(z_hat)

        mu_y = y_para[:, 0:12]
        sigma_y = 0.01 + torch.abs(y_para[:, 12:24])

        y_res = y - mu_y
        y_hat_res = compression_encoder.integer_quantization(y_res)
        y_hat = y_hat_res + mu_y

        # Décodage
        x_hat = compression_decoder(y_hat)

    # Calculer la taille de l'image compressée
    spatial_px = x.shape[2] * x.shape[3]
    rate_loss_y = losses.rate_loss(y, mu_y, sigma_y)
    rate_loss_z = losses.rate_loss(z)
    bpp_y = rate_loss_y / spatial_px
    bpp_z = rate_loss_z / spatial_px
    bpp_total = bpp_y + bpp_z

    # Calculer le taux de compression implicite
    compression_ratio = 24 / bpp_total.item()  # 24 bits par pixel (8 bits par canal RGB)

    # Convertir l'image reconstruite en format PIL
    x_hat_pil = transforms.ToPILImage()(x_hat.squeeze().cpu())

    # Sauvegarder l'image reconstruite
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    x_hat_pil.save(output_path)

    # Afficher les images et les métriques
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image originale")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(x_hat_pil)
    plt.title(f"Image compressée\nBpp: {bpp_total.item():.2f}, Ratio: {compression_ratio:.2f}:1")
    plt.axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "comparison_" + os.path.basename(image_path))
    plt.savefig(comparison_path)
    plt.show()

    print(f"Image originale: {image_path}")
    print(f"Image compressée: {output_path}")
    print(f"Comparaison: {comparison_path}")
    print(f"Bits par pixel (bpp): {bpp_total.item():.4f}")
    print(f"Taux de compression: {compression_ratio:.2f}:1")

    return x_hat_pil, bpp_total.item(), compression_ratio


def main():
    # Configuration de l'analyseur d'arguments pour l'interface en ligne de commande
    parser = argparse.ArgumentParser(description='Compression d\'image avec IA')
    parser.add_argument('--image', type=str, required=True, help='Chemin vers l\'image à compresser')
    parser.add_argument('--output', type=str, default='compressed_output',
                        help='Répertoire de sortie pour les images compressées')
    parser.add_argument('--model', type=str, default='saved_models/compression_model_2000000.pt',
                        help='Chemin vers le modèle de compression')

    args = parser.parse_args()

    # Vérifier si le fichier modèle existe et est au format .pt
    if not os.path.exists(args.model):
        # Si le fichier .pt n'existe pas, vérifier si un fichier .rar avec le même nom existe
        rar_path = args.model.replace('.pt', '.rar')
        if os.path.exists(rar_path):
            print(f"Le fichier modèle est au format .rar ({rar_path})")
            print("Veuillez extraire le fichier RAR avant de continuer.")
            print("Utilisez un utilitaire comme WinRAR, 7-Zip ou la commande 'unrar' pour extraire le fichier.")
            return
        else:
            print(f"Le fichier modèle {args.model} n'existe pas.")
            return

    # Vérifier si le chemin de l'image est valide
    if not os.path.exists(args.image):
        print(f"L'image {args.image} n'existe pas.")
        return

    # Compresser l'image
    compress_image(args.image, args.output, args.model)


if __name__ == '__main__':
    main()