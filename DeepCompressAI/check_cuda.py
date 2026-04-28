#!/usr/bin/env python3
"""
Script de vérification et optimisation CUDA pour DeepCompress
"""

import torch
import time
import psutil
import subprocess

def check_cuda_setup():
    """Vérifie la configuration CUDA complète"""
    print("=" * 60)
    print("        VÉRIFICATION CONFIGURATION CUDA")
    print("=" * 60)
    
    # Vérification PyTorch CUDA
    print("1. Configuration PyTorch:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   Version CUDA: {torch.version.cuda}")
        print(f"   Nombre de GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name}")
            print(f"      Mémoire totale: {props.total_memory / 1024**3:.1f} GB")
            print(f"      Compute capability: {props.major}.{props.minor}")
    
    # Vérification drivers NVIDIA
    print("\n2. Drivers NVIDIA:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"   ✓ {line.strip()}")
                if 'RTX 4050' in line or 'GeForce' in line:
                    print(f"   ✓ GPU détecté: {line.strip()}")
        else:
            print("   ✗ nvidia-smi non accessible")
    except FileNotFoundError:
        print("   ✗ nvidia-smi non trouvé (drivers NVIDIA probablement manquants)")
    
    return torch.cuda.is_available()

def benchmark_inference():
    """Teste les performances d'inférence sur CPU vs GPU"""
    print("\n3. Benchmark d'inférence:")
    
    if not torch.cuda.is_available():
        print("   ✗ CUDA non disponible, benchmark impossible")
        return
    
    try:
        from src import models
        
        # Test sur CPU
        print("   Test CPU...")
        device_cpu = torch.device('cpu')
        encoder_cpu = models.Encoder().to(device_cpu)
        
        # Image test
        test_input = torch.randn(1, 3, 256, 256).to(device_cpu)
        
        start_time = time.time()
        with torch.no_grad():
            _ = encoder_cpu(test_input)
        cpu_time = time.time() - start_time
        print(f"   Temps CPU: {cpu_time:.3f}s")
        
        # Test sur GPU
        print("   Test GPU...")
        device_gpu = torch.device('cuda')
        encoder_gpu = models.Encoder().to(device_gpu)
        test_input_gpu = test_input.to(device_gpu)
        
        # Warmup GPU
        with torch.no_grad():
            _ = encoder_gpu(test_input_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            _ = encoder_gpu(test_input_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"   Temps GPU: {gpu_time:.3f}s")
        print(f"   Accélération: {cpu_time/gpu_time:.1f}x")
        
        # Mémoire GPU
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"   Mémoire GPU utilisée: {memory_allocated:.1f}MB")
        print(f"   Mémoire GPU réservée: {memory_reserved:.1f}MB")
        
    except Exception as e:
        print(f"   ✗ Erreur lors du benchmark: {e}")

def check_system_resources():
    """Vérifie les ressources système"""
    print("\n4. Ressources système:")
    print(f"   RAM totale: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"   RAM disponible: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print(f"   CPU cores: {psutil.cpu_count()}")

def main():
    cuda_available = check_cuda_setup()
    check_system_resources()
    
    if cuda_available:
        benchmark_inference()
        print("\n" + "=" * 60)
        print("✅ CUDA CONFIGURÉ - Optimisations recommandées:")
        print("   1. Utiliser des batch sizes plus grands")
        print("   2. Activer mixed precision (AMP)")
        print("   3. Optimiser les transferts mémoire")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ CUDA NON DISPONIBLE")
        print("   Vérifiez l'installation des drivers NVIDIA")
        print("   et la version CUDA compatible avec PyTorch")
        print("=" * 60)

if __name__ == '__main__':
    main()