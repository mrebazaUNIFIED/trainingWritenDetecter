
from doctr.models import ocr_predictor
import torch
import os

def download_base_model():
    print("📥 Descargando modelo base de docTR...")
    
    # Descargar modelo completo de OCR (detector + reconocedor)
    model = ocr_predictor(
        det_arch='db_resnet50',      # Detector de texto
        reco_arch='crnn_vgg16_bn',   # Reconocedor de texto
        pretrained=True
    )
    
    print("✅ Modelo descargado exitosamente")
    
    # Guardar componentes por separado
    os.makedirs('models/pretrained', exist_ok=True)
    
    # Guardar detector
    torch.save(
        model.det_predictor.model.state_dict(),
        'models/pretrained/detector_base.pth'
    )
    print("💾 Detector guardado en models/pretrained/detector_base.pth")
    
    # Guardar reconocedor
    torch.save(
        model.reco_predictor.model.state_dict(),
        'models/pretrained/recognizer_base.pth'
    )
    print("💾 Reconocedor guardado en models/pretrained/recognizer_base.pth")
    
    # Probar el modelo
    print("\n🧪 Probando modelo con una imagen de prueba...")
    test_prediction(model)
    
    return model

def test_prediction(model):
    """Prueba rápida del modelo"""
    from doctr.io import DocumentFile
    import numpy as np
    from PIL import Image
    
    # Crear imagen de prueba simple
    img = Image.new('RGB', (400, 100), color='white')
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    try:
        # Intentar usar fuente del sistema
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 30), "Prueba OCR", fill='black', font=font)
    
    # Guardar imagen temporal
    img.save('test_image.png')
    
    # Hacer predicción
    doc = DocumentFile.from_images('test_image.png')
    result = model(doc)
    
    # Mostrar resultado
    print("\n📄 Resultado de la predicción:")
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    print(f"  - Texto detectado: '{word.value}' (confianza: {word.confidence:.2f})")
    
    # Limpiar
    os.remove('test_image.png')

if __name__ == "__main__":
    model = download_base_model()
    print("\n✅ ¡Listo! Modelo base descargado y guardado.")