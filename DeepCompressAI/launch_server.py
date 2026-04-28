#!/usr/bin/env python3
"""
Script de lancement pour l'application DeepCompress
Ce script vérifie les dépendances et lance le serveur Flask
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path


def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    required_packages = ['torch', 'torchvision', 'flask', 'flask_cors', 'PIL', 'numpy']
    missing_packages = []

    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    return missing_packages


def check_model_file():
    """Vérifie la présence du fichier modèle"""
    model_path = "saved_models/compression_model_2000000.pt"
    rar_path = "saved_models/compression_model_2000000.rar"

    if os.path.exists(model_path):
        return True, "Fichier modèle trouvé"
    elif os.path.exists(rar_path):
        return False, f"Le modèle est au format RAR ({rar_path}). Veuillez l'extraire avant de continuer."
    else:
        return False, "Fichier modèle non trouvé"


def install_dependencies():
    """Installe les dépendances manquantes"""
    print("Installation des dépendances...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    print("=" * 60)
    print("        DeepCompress - Serveur de Compression IA")
    print("=" * 60)

    # Vérifier les dépendances
    print("1. Vérification des dépendances...")
    missing = check_dependencies()

    if missing:
        print(f"   Dépendances manquantes: {', '.join(missing)}")

        if os.path.exists("requirements.txt"):
            response = input("   Voulez-vous les installer automatiquement ? (o/n): ")
            if response.lower() in ['o', 'oui', 'y', 'yes']:
                if install_dependencies():
                    print("   ✓ Dépendances installées avec succès")
                else:
                    print("   ✗ Erreur lors de l'installation")
                    return
            else:
                print("   Veuillez installer les dépendances manuellement:")
                print("   pip install -r requirements.txt")
                return
        else:
            print("   Fichier requirements.txt non trouvé")
            return
    else:
        print("   ✓ Toutes les dépendances sont présentes")

    # Vérifier le fichier modèle
    print("2. Vérification du modèle...")
    model_ok, model_msg = check_model_file()
    print(f"   {model_msg}")

    if not model_ok:
        print("   ✗ Impossible de continuer sans le modèle")
        return
    else:
        print("   ✓ Modèle prêt")

    # Vérifier la structure du projet
    print("3. Vérification de la structure du projet...")
    required_files = ['src/__init__.py', 'src/models.py', 'src/utils.py', 'src/losses.py', 'inference.html']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"   ✗ Fichiers manquants: {', '.join(missing_files)}")
        return
    else:
        print("   ✓ Structure du projet correcte")

    print("\n4. Démarrage du serveur...")
    print("   Le serveur va démarrer sur http://localhost:5000")
    print("   L'interface web s'ouvrira automatiquement dans votre navigateur")
    print("\n   Pour arrêter le serveur, appuyez sur Ctrl+C")
    print("=" * 60)

    # Démarrer le serveur
    try:
        # Ouvrir le navigateur après un court délai
        import threading
        def open_browser():
            import time
            time.sleep(2)  # Attendre que le serveur démarre
            webbrowser.open('http://localhost:5000')

        threading.Thread(target=open_browser).start()

        # Importer et lancer l'application Flask
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)

    except KeyboardInterrupt:
        print("\n\nServeur arrêté par l'utilisateur")
    except Exception as e:
        print(f"\nErreur lors du démarrage: {e}")


if __name__ == '__main__':
    main()