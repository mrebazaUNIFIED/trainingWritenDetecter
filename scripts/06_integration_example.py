"""
Ejemplo de cómo integrar el modelo OCR entrenado en tu otro proyecto
Copia este archivo a tu proyecto principal
"""
import torch
from doctr.models import recognition, detection, ocr_predictor
from doctr.io import DocumentFile
import json
import os
from PIL import Image

class CustomOCR:
    """
    Clase para usar tu modelo OCR entrenado en cualquier proyecto
    """
    
    def __init__(self, model_path, vocab_path=None, device='auto'):
        """
        Inicializa el OCR con tu modelo entrenado
        
        Args:
            model_path: Ruta al archivo .pth del modelo entrenado
            vocab_path: Ruta al vocab.json (opcional si está en el checkpoint)
            device: 'cuda', 'cpu', o 'auto'
        """
        # Configurar device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️ Dispositivo: {self.device}")
        
        # Cargar checkpoint
        print(f"📦 Cargando modelo desde {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Obtener vocabulario
        if 'vocab' in checkpoint:
            self.vocab = checkpoint['vocab']
            print("✅ Vocabulario cargado desde checkpoint")
        elif vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            print("✅ Vocabulario cargado desde archivo")
        else:
            raise ValueError("No se encontró el vocabulario")
        
        # Crear modelo de reconocimiento
        self.recognition_model = recognition.crnn_vgg16_bn(
            pretrained=False,
            vocab=self.vocab
        )
        self.recognition_model.load_state_dict(checkpoint['model_state_dict'])
        self.recognition_model = self.recognition_model.to(self.device)
        self.recognition_model.eval()
        
        # Crear predictor completo (detector + reconocedor)
        self.predictor = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch=self.recognition_model,
            pretrained=True
        )
        
        print("✅ Modelo cargado exitosamente")
    
    def process_image(self, image_source):
        """
        Procesa una imagen y extrae todo el texto
        
        Args:
            image_source: puede ser:
                - str: ruta a archivo de imagen
                - PIL.Image: imagen PIL
                - numpy.ndarray: array numpy
        
        Returns:
            dict con texto y metadatos
        """
        # Cargar documento
        if isinstance(image_source, str):
            doc = DocumentFile.from_images(image_source)
        else:
            doc = DocumentFile.from_images([image_source])
        
        # Hacer predicción
        result = self.predictor(doc)
        
        # Extraer texto
        full_text = self._extract_text_from_result(result)
        
        # Extraer detalles
        details = self._extract_details_from_result(result)
        
        return {
            'text': full_text,
            'details': details,
            'num_pages': len(result.pages),
            'num_words': sum(len(d['words']) for d in details['pages'])
        }
    
    def _extract_text_from_result(self, result):
        """Extrae texto completo del resultado"""
        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = [word.value for word in line.words]
                    lines.append(' '.join(words))
        
        return '\n'.join(lines)
    
    def _extract_details_from_result(self, result):
        """Extrae detalles completos incluyendo coordenadas"""
        pages_data = []
        
        for page in result.pages:
            page_data = {
                'dimensions': page.dimensions,
                'blocks': []
            }
            
            for block in page.blocks:
                block_data = {
                    'geometry': block.geometry,
                    'lines': []
                }
                
                for line in block.lines:
                    line_data = {
                        'geometry': line.geometry,
                        'words': []
                    }
                    
                    for word in line.words:
                        word_data = {
                            'value': word.value,
                            'confidence': word.confidence,
                            'geometry': word.geometry
                        }
                        line_data['words'].append(word_data)
                    
                    block_data['lines'].append(line_data)
                
                page_data['blocks'].append(block_data)
            
            pages_data.append(page_data)
        
        return {'pages': pages_data}
    
    def process_batch(self, image_paths):
        """
        Procesa múltiples imágenes
        
        Args:
            image_paths: lista de rutas a imágenes
        
        Returns:
            lista de resultados
        """
        results = []
        
        for img_path in image_paths:
            try:
                result = self.process_image(img_path)
                result['image'] = img_path
                result['status'] = 'success'
                results.append(result)
            except Exception as e:
                results.append({
                    'image': img_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def get_text_with_confidence(self, image_source, min_confidence=0.5):
        """
        Extrae texto filtrado por nivel de confianza
        
        Args:
            image_source: imagen a procesar
            min_confidence: confianza mínima (0-1)
        
        Returns:
            dict con texto y palabras filtradas
        """
        result = self.process_image(image_source)
        
        filtered_words = []
        
        for page in result['details']['pages']:
            for block in page['blocks']:
                for line in block['lines']:
                    for word in line['words']:
                        if word['confidence'] >= min_confidence:
                            filtered_words.append({
                                'text': word['value'],
                                'confidence': word['confidence'],
                                'position': word['geometry']
                            })
        
        filtered_text = ' '.join([w['text'] for w in filtered_words])
        
        return {
            'text': filtered_text,
            'words': filtered_words,
            'total_words': len(filtered_words),
            'avg_confidence': sum(w['confidence'] for w in filtered_words) / len(filtered_words) if filtered_words else 0
        }
    
    def extract_region(self, image_source, bbox):
        """
        Extrae texto de una región específica de la imagen
        
        Args:
            image_source: imagen a procesar
            bbox: (x1, y1, x2, y2) coordenadas normalizadas (0-1)
        
        Returns:
            texto de la región
        """
        result = self.process_image(image_source)
        
        x1, y1, x2, y2 = bbox
        region_text = []
        
        for page in result['details']['pages']:
            for block in page['blocks']:
                for line in block['lines']:
                    for word in line['words']:
                        # Verificar si la palabra está en la región
                        word_bbox = word['geometry']
                        wx1, wy1 = word_bbox[0]
                        wx2, wy2 = word_bbox[2]
                        
                        # Check overlap
                        if (wx1 >= x1 and wx2 <= x2 and 
                            wy1 >= y1 and wy2 <= y2):
                            region_text.append(word['value'])
        
        return ' '.join(region_text)

# ==================== EJEMPLOS DE USO ====================

def ejemplo_basico():
    """Ejemplo básico de uso"""
    print("\n" + "="*70)
    print("EJEMPLO 1: USO BÁSICO")
    print("="*70)
    
    # Inicializar OCR con tu modelo
    ocr = CustomOCR(
        model_path='models/final/production_model.pth'
    )
    
    # Procesar una imagen
    result = ocr.process_image('test_image.jpg')
    
    print(f"\n📄 Texto extraído:")
    print(result['text'])
    
    print(f"\n📊 Estadísticas:")
    print(f"   Páginas: {result['num_pages']}")
    print(f"   Palabras: {result['num_words']}")

def ejemplo_con_confianza():
    """Ejemplo filtrando por confianza"""
    print("\n" + "="*70)
    print("EJEMPLO 2: FILTRADO POR CONFIANZA")
    print("="*70)
    
    ocr = CustomOCR(model_path='models/final/production_model.pth')
    
    # Obtener solo texto con alta confianza
    result = ocr.get_text_with_confidence(
        'test_image.jpg',
        min_confidence=0.8  # 80% confianza mínima
    )
    
    print(f"\n📄 Texto (confianza ≥ 80%):")
    print(result['text'])
    
    print(f"\n📊 Estadísticas:")
    print(f"   Palabras detectadas: {result['total_words']}")
    print(f"   Confianza promedio: {result['avg_confidence']:.2%}")
    
    # Mostrar palabras con baja confianza
    low_conf = [w for w in result['words'] if w['confidence'] < 0.9]
    if low_conf:
        print(f"\n⚠️ Palabras con confianza < 90%:")
        for w in low_conf:
            print(f"   '{w['text']}' - {w['confidence']:.2%}")

def ejemplo_batch():
    """Ejemplo procesando múltiples imágenes"""
    print("\n" + "="*70)
    print("EJEMPLO 3: PROCESAMIENTO BATCH")
    print("="*70)
    
    ocr = CustomOCR(model_path='models/final/production_model.pth')
    
    # Lista de imágenes a procesar
    images = [
        'invoice_001.jpg',
        'invoice_002.jpg',
        'receipt_001.jpg'
    ]
    
    # Procesar todas
    results = ocr.process_batch(images)
    
    # Mostrar resultados
    for result in results:
        print(f"\n📄 {result['image']}")
        if result['status'] == 'success':
            print(f"   ✅ Texto: {result['text'][:100]}...")
            print(f"   📊 Palabras: {result['num_words']}")
        else:
            print(f"   ❌ Error: {result['error']}")

def ejemplo_region_especifica():
    """Ejemplo extrayendo texto de región específica"""
    print("\n" + "="*70)
    print("EJEMPLO 4: EXTRACCIÓN DE REGIÓN ESPECÍFICA")
    print("="*70)
    
    ocr = CustomOCR(model_path='models/final/production_model.pth')
    
    # Extraer solo la región superior izquierda (25% de la imagen)
    region_text = ocr.extract_region(
        'invoice.jpg',
        bbox=(0, 0, 0.5, 0.25)  # x1, y1, x2, y2 normalizados
    )
    
    print(f"\n📄 Texto de región superior izquierda:")
    print(region_text)

def ejemplo_integracion_api():
    """Ejemplo de integración en una API"""
    print("\n" + "="*70)
    print("EJEMPLO 5: INTEGRACIÓN EN API (FastAPI)")
    print("="*70)
    
    print("""
# Ejemplo de código para FastAPI:

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

# Inicializar OCR una sola vez al arrancar
ocr = CustomOCR(model_path='models/final/production_model.pth')

@app.post("/ocr/process")
async def process_ocr(file: UploadFile = File(...)):
    '''Endpoint para procesar OCR'''
    
    # Leer imagen
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Procesar con OCR
    result = ocr.process_image(image)
    
    return {
        "status": "success",
        "text": result['text'],
        "num_words": result['num_words'],
        "confidence": result.get('avg_confidence', 0)
    }

@app.post("/ocr/batch")
async def process_batch_ocr(files: list[UploadFile] = File(...)):
    '''Endpoint para procesamiento batch'''
    
    results = []
    
    for file in files:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        result = ocr.process_image(image)
        
        results.append({
            "filename": file.filename,
            "text": result['text'],
            "num_words": result['num_words']
        })
    
    return {"status": "success", "results": results}

# Ejecutar: uvicorn main:app --reload
    """)

def ejemplo_flask():
    """Ejemplo de integración en Flask"""
    print("\n" + "="*70)
    print("EJEMPLO 6: INTEGRACIÓN EN FLASK")
    print("="*70)
    
    print("""
# Ejemplo de código para Flask:

from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Inicializar OCR
ocr = CustomOCR(model_path='models/final/production_model.pth')

@app.route('/ocr', methods=['POST'])
def process_ocr():
    '''Endpoint para procesar OCR'''
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Leer imagen
    image = Image.open(io.BytesIO(file.read()))
    
    # Procesar
    result = ocr.process_image(image)
    
    return jsonify({
        'status': 'success',
        'text': result['text'],
        'num_words': result['num_words']
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# Ejecutar: python app.py
    """)

def ejemplo_streamlit():
    """Ejemplo de aplicación web con Streamlit"""
    print("\n" + "="*70)
    print("EJEMPLO 7: APLICACIÓN WEB CON STREAMLIT")
    print("="*70)
    
    print("""
# Ejemplo de código para Streamlit:

import streamlit as st
from PIL import Image

# Inicializar OCR (solo una vez)
@st.cache_resource
def load_ocr():
    return CustomOCR(model_path='models/final/production_model.pth')

ocr = load_ocr()

# Interfaz
st.title("🔍 OCR - Extractor de Texto")

uploaded_file = st.file_uploader(
    "Sube una imagen", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    # Mostrar imagen
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_column_width=True)
    
    # Procesar
    with st.spinner('Procesando...'):
        result = ocr.process_image(image)
    
    # Mostrar resultados
    st.success('✅ Procesamiento completado')
    
    st.subheader("📄 Texto extraído:")
    st.text_area("Resultado", result['text'], height=200)
    
    st.subheader("📊 Estadísticas:")
    col1, col2 = st.columns(2)
    col1.metric("Palabras detectadas", result['num_words'])
    col2.metric("Páginas", result['num_pages'])

# Ejecutar: streamlit run app.py
    """)

# ==================== SCRIPT PRINCIPAL ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EJEMPLOS DE INTEGRACIÓN DEL MODELO OCR")
    print("="*70)
    
    print("\n📚 Ejemplos disponibles:")
    print("  1. Uso básico")
    print("  2. Filtrado por confianza")
    print("  3. Procesamiento batch")
    print("  4. Extracción de región específica")
    print("  5. Integración en API (FastAPI)")
    print("  6. Integración en Flask")
    print("  7. Aplicación web con Streamlit")
    print("  8. Ejecutar todos los ejemplos")
    
    opcion = input("\nSelecciona un ejemplo (1-8): ").strip()
    
    ejemplos = {
        '1': ejemplo_basico,
        '2': ejemplo_con_confianza,
        '3': ejemplo_batch,
        '4': ejemplo_region_especifica,
        '5': ejemplo_integracion_api,
        '6': ejemplo_flask,
        '7': ejemplo_streamlit,
    }
    
    if opcion in ejemplos:
        ejemplos[opcion]()
    elif opcion == '8':
        for func in ejemplos.values():
            func()
    else:
        print("❌ Opción inválida")