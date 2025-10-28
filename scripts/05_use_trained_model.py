"""
Script para usar el modelo OCR entrenado en nuevas imágenes
"""
import torch
from doctr.models import recognition, ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import json
import os
import sys

class TrainedOCRModel:
    """Wrapper para el modelo OCR entrenado"""
    
    def __init__(self, model_path, vocab_path, device='cuda'):
        """
        Carga el modelo entrenado
        
        Args:
            model_path: Ruta al checkpoint del modelo (.pth)
            vocab_path: Ruta al vocabulario (vocab.json)
            device: 'cuda' o 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Usando dispositivo: {self.device}")
        
        # Cargar vocabulario
        print(f"📚 Cargando vocabulario desde {vocab_path}...")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        print(f"   Vocabulario: {len(self.vocab)} caracteres")
        
        # Cargar modelo
        print(f"🤖 Cargando modelo desde {model_path}...")
        self.model = recognition.crnn_vgg16_bn(
            pretrained=False,
            vocab=self.vocab
        )
        
        # Cargar pesos entrenados
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Modelo cargado exitosamente")
        print(f"   Época de entrenamiento: {checkpoint['epoch']}")
        print(f"   Loss de validación: {checkpoint['loss']:.4f}")
    
    def predict_image(self, image_path):
        """
        Predice texto en una imagen
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            str: Texto detectado
        """
        # Cargar imagen
        image = Image.open(image_path).convert('RGB')
        
        # Hacer predicción
        with torch.no_grad():
            # Preprocesar imagen (ajustar según tu modelo)
            from doctr import transforms as T
            transform = T.Compose([
                T.Resize((32, 128))
            ])
            
            img_tensor = transform(image)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Predicción
            output = self.model(img_tensor)
            
            # Decodificar (simplificado - ajustar según tu implementación)
            prediction = self._decode_output(output)
        
        return prediction
    
    def _decode_output(self, output):
        """Decodifica la salida del modelo a texto"""
        # Implementación simplificada
        # Ajustar según la arquitectura exacta de tu modelo
        pred_indices = output.argmax(dim=-1)
        
        # Convertir índices a caracteres
        chars = []
        prev_char = None
        
        for idx in pred_indices[0]:
            idx = idx.item()
            if idx == 0:  # padding
                continue
            
            char = self.vocab[idx] if idx < len(self.vocab) else ''
            
            # Evitar caracteres repetidos (CTC decoding)
            if char != prev_char and char not in ['<pad>', '<sos>', '<eos>']:
                chars.append(char)
            
            prev_char = char
        
        return ''.join(chars)

def create_ocr_predictor_with_trained_model(model_path, vocab_path):
    """
    Crea un predictor de OCR completo usando el modelo entrenado
    
    Esto combina detección (modelo base) + reconocimiento (tu modelo entrenado)
    """
    print("🔧 Creando predictor OCR completo...")
    
    # Cargar vocabulario
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # Crear modelo de reconocimiento con tus pesos
    reco_model = recognition.crnn_vgg16_bn(pretrained=False, vocab=vocab)
    checkpoint = torch.load(model_path, map_location='cpu')
    reco_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Crear predictor completo (detector base + tu reconocedor)
    predictor = ocr_predictor(
        det_arch='db_resnet50',  # Detector de texto (base)
        reco_arch=reco_model,     # Tu reconocedor entrenado
        pretrained=True
    )
    
    print("✅ Predictor OCR listo")
    return predictor

# ==================== FUNCIONES DE USO ====================

def predict_single_image(model_path, vocab_path, image_path, output_dir='outputs/predictions'):
    """
    Predice texto en una sola imagen
    """
    print("\n" + "="*70)
    print("PREDICCIÓN EN IMAGEN INDIVIDUAL")
    print("="*70)
    
    # Cargar modelo
    ocr = TrainedOCRModel(model_path, vocab_path)
    
    # Predecir
    print(f"\n📷 Procesando: {image_path}")
    result = ocr.predict_image(image_path)
    
    print(f"\n📝 Texto detectado:")
    print(f"   {result}")
    
    # Guardar resultado
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'prediction.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Imagen: {image_path}\n")
        f.write(f"Texto detectado: {result}\n")
    
    print(f"\n💾 Resultado guardado en: {output_file}")
    
    return result

def predict_batch(model_path, vocab_path, images_folder, output_dir='outputs/predictions'):
    """
    Predice texto en múltiples imágenes
    """
    print("\n" + "="*70)
    print("PREDICCIÓN EN BATCH")
    print("="*70)
    
    # Cargar modelo
    ocr = TrainedOCRModel(model_path, vocab_path)
    
    # Listar imágenes
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    image_files = [
        f for f in os.listdir(images_folder)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    print(f"\n📁 Encontradas {len(image_files)} imágenes")
    
    # Procesar cada imagen
    results = []
    
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(images_folder, img_file)
        
        print(f"\n[{idx}/{len(image_files)}] Procesando: {img_file}")
        
        try:
            prediction = ocr.predict_image(img_path)
            print(f"   ✅ Texto: {prediction}")
            
            results.append({
                'image': img_file,
                'text': prediction,
                'status': 'success'
            })
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'image': img_file,
                'text': '',
                'status': 'error',
                'error': str(e)
            })
    
    # Guardar resultados
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar en JSON
    output_json = os.path.join(output_dir, 'batch_results.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Guardar en TXT
    output_txt = os.path.join(output_dir, 'batch_results.txt')
    with open(output_txt, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f"Imagen: {r['image']}\n")
            f.write(f"Texto: {r['text']}\n")
            f.write(f"Estado: {r['status']}\n")
            f.write("-" * 50 + "\n")
    
    print(f"\n💾 Resultados guardados en:")
    print(f"   - {output_json}")
    print(f"   - {output_txt}")
    
    # Estadísticas
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\n📊 Estadísticas:")
    print(f"   Total: {len(results)}")
    print(f"   Exitosas: {successful}")
    print(f"   Errores: {len(results) - successful}")
    
    return results

def export_model_for_production(model_path, vocab_path, output_path='models/final/production_model.pth'):
    """
    Exporta el modelo en formato optimizado para producción
    """
    print("\n" + "="*70)
    print("EXPORTANDO MODELO PARA PRODUCCIÓN")
    print("="*70)
    
    # Cargar modelo
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    model = recognition.crnn_vgg16_bn(pretrained=False, vocab=vocab)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Crear directorio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar modelo optimizado
    export_dict = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'model_config': {
            'architecture': 'crnn_vgg16_bn',
            'input_size': (32, 128),
            'vocab_size': len(vocab)
        },
        'training_info': {
            'final_epoch': checkpoint['epoch'],
            'final_loss': checkpoint['loss']
        }
    }
    
    torch.save(export_dict, output_path)
    
    print(f"✅ Modelo exportado exitosamente")
    print(f"📦 Ubicación: {output_path}")
    print(f"💾 Tamaño: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Crear archivo README
    readme_path = os.path.join(os.path.dirname(output_path), 'README.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("MODELO OCR ENTRENADO - INSTRUCCIONES DE USO\n")
        f.write("=" * 70 + "\n\n")
        f.write("Este modelo está listo para usar en producción.\n\n")
        f.write("Para cargarlo en tu proyecto:\n\n")
        f.write("```python\n")
        f.write("import torch\n")
        f.write("from doctr.models import recognition\n\n")
        f.write(f"# Cargar modelo\n")
        f.write(f"checkpoint = torch.load('{output_path}')\n")
        f.write("vocab = checkpoint['vocab']\n")
        f.write("model = recognition.crnn_vgg16_bn(pretrained=False, vocab=vocab)\n")
        f.write("model.load_state_dict(checkpoint['model_state_dict'])\n")
        f.write("model.eval()\n")
        f.write("```\n")
    
    print(f"📄 README creado: {readme_path}")
    
    return output_path

# ==================== MENÚ PRINCIPAL ====================

if __name__ == "__main__":
    # Rutas por defecto
    MODEL_PATH = 'models/checkpoints/best_model.pth'
    VOCAB_PATH = 'models/checkpoints/vocab.json'
    
    # Verificar que existe el modelo
    if not os.path.exists(MODEL_PATH):
        print("❌ No se encontró el modelo entrenado")
        print(f"   Buscado en: {MODEL_PATH}")
        print("\n💡 Asegúrate de haber entrenado el modelo primero:")
        print("   python scripts/04_train_model.py")
        sys.exit(1)
    
    if not os.path.exists(VOCAB_PATH):
        print("❌ No se encontró el vocabulario")
        print(f"   Buscado en: {VOCAB_PATH}")
        sys.exit(1)
    
    # Menú interactivo
    print("\n" + "="*70)
    print("USO DEL MODELO OCR ENTRENADO")
    print("="*70)
    print("\nOpciones:")
    print("  1. Predecir en una imagen")
    print("  2. Predecir en carpeta de imágenes (batch)")
    print("  3. Exportar modelo para producción")
    print("  4. Salir")
    
    while True:
        print("\n" + "-"*70)
        opcion = input("Selecciona una opción (1-4): ").strip()
        
        if opcion == '1':
            imagen = input("Ruta de la imagen: ").strip()
            if os.path.exists(imagen):
                predict_single_image(MODEL_PATH, VOCAB_PATH, imagen)
            else:
                print("❌ Imagen no encontrada")
        
        elif opcion == '2':
            carpeta = input("Ruta de la carpeta con imágenes: ").strip()
            if os.path.exists(carpeta):
                predict_batch(MODEL_PATH, VOCAB_PATH, carpeta)
            else:
                print("❌ Carpeta no encontrada")
        
        elif opcion == '3':
            output = input("Ruta de salida (Enter para default): ").strip()
            if not output:
                output = 'models/final/production_model.pth'
            export_model_for_production(MODEL_PATH, VOCAB_PATH, output)
        
        elif opcion == '4':
            print("👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción inválida")