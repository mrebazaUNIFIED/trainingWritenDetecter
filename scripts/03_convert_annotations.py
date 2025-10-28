"""
Script para convertir anotaciones de Label Studio al formato docTR
"""
import json
import os
from pathlib import Path
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split

def convert_labelstudio_to_doctr(annotations_file, images_folder, output_folder):
    """
    Convierte anotaciones de Label Studio al formato docTR
    """
    print("🔄 Convirtiendo anotaciones...")
    
    # Cargar anotaciones
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"📊 Total de imágenes anotadas: {len(annotations)}")
    
    # Preparar estructura
    os.makedirs(f"{output_folder}/images", exist_ok=True)
    
    dataset = []
    skipped = 0
    
    for idx, item in enumerate(annotations):
        try:
            # Obtener nombre de archivo de imagen
            image_filename = item['data']['image'].split('/')[-1]
            image_path = os.path.join(images_folder, image_filename)
            
            if not os.path.exists(image_path):
                print(f"⚠️ Imagen no encontrada: {image_filename}")
                skipped += 1
                continue
            
            # Obtener dimensiones de la imagen
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Procesar anotaciones
            if 'annotations' not in item or len(item['annotations']) == 0:
                print(f"⚠️ Sin anotaciones: {image_filename}")
                skipped += 1
                continue
            
            words = []
            
            for annotation in item['annotations'][0]['result']:
                if annotation['type'] == 'rectanglelabels' or annotation['type'] == 'rectangle':
                    # Obtener coordenadas del bounding box
                    bbox = annotation['value']
                    
                    # Convertir porcentajes a coordenadas absolutas
                    x_min = bbox['x'] * img_width / 100
                    y_min = bbox['y'] * img_height / 100
                    width = bbox['width'] * img_width / 100
                    height = bbox['height'] * img_height / 100
                    
                    x_max = x_min + width
                    y_max = y_min + height
                    
                    # Normalizar coordenadas (0-1)
                    x_min_norm = x_min / img_width
                    y_min_norm = y_min / img_height
                    x_max_norm = x_max / img_width
                    y_max_norm = y_max / img_height
                    
                    # Buscar la transcripción correspondiente
                    transcription = ""
                    text_type = "digital"
                    
                    # Buscar en todos los resultados de esta anotación
                    annotation_id = annotation.get('id', '')
                    
                    for result in item['annotations'][0]['result']:
                        # Buscar textarea con el mismo ID o parent_id
                        if result['type'] == 'textarea':
                            if result.get('from_name') == 'transcription':
                                # Verificar si corresponde a este bbox
                                transcription = result['value'].get('text', [''])[0]
                                break
                        
                        # Buscar tipo de texto
                        if result['type'] == 'choices' and result.get('from_name') == 'text_type':
                            text_type = result['value'].get('choices', ['digital'])[0]
                    
                    if not transcription:
                        print(f"⚠️ Bbox sin transcripción en {image_filename}")
                        continue
                    
                    # Agregar palabra al dataset
                    words.append({
                        'geometry': [
                            [x_min_norm, y_min_norm],
                            [x_max_norm, y_min_norm],
                            [x_max_norm, y_max_norm],
                            [x_min_norm, y_max_norm]
                        ],
                        'value': transcription,
                        'type': text_type
                    })
            
            if len(words) == 0:
                print(f"⚠️ No se encontraron palabras válidas en {image_filename}")
                skipped += 1
                continue
            
            # Copiar imagen a carpeta de salida
            new_image_name = f"img_{idx:05d}{Path(image_path).suffix}"
            new_image_path = f"{output_folder}/images/{new_image_name}"
            shutil.copy2(image_path, new_image_path)
            
            # Agregar al dataset
            dataset.append({
                'image': new_image_name,
                'words': words
            })
            
            if (idx + 1) % 50 == 0:
                print(f"  Procesadas {idx + 1} imágenes...")
        
        except Exception as e:
            print(f"❌ Error procesando {item.get('data', {}).get('image', 'unknown')}: {e}")
            skipped += 1
    
    print(f"\n✅ Procesadas: {len(dataset)} imágenes")
    print(f"⚠️ Omitidas: {skipped} imágenes")
    
    # Guardar dataset completo
    with open(f"{output_folder}/dataset.json", 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Dataset guardado en {output_folder}/dataset.json")
    
    return dataset

def split_dataset(dataset, output_base='data/splits', train_ratio=0.7, val_ratio=0.15):
    """
    Divide el dataset en train/val/test
    """
    print(f"\n📊 Dividiendo dataset...")
    print(f"   Train: {train_ratio*100}%")
    print(f"   Val: {val_ratio*100}%")
    print(f"   Test: {(1-train_ratio-val_ratio)*100}%")
    
    # Dividir primero train y temp (val+test)
    train_data, temp_data = train_test_split(
        dataset, 
        train_size=train_ratio, 
        random_state=42
    )
    
    # Dividir temp en val y test
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_data, test_data = train_test_split(
        temp_data, 
        train_size=val_ratio_adjusted, 
        random_state=42
    )
    
    print(f"\n📈 Tamaños de splits:")
    print(f"   Train: {len(train_data)} imágenes")
    print(f"   Val: {len(val_data)} imágenes")
    print(f"   Test: {len(test_data)} imágenes")
    
    # Guardar cada split
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        split_dir = f"{output_base}/{split_name}"
        os.makedirs(f"{split_dir}/images", exist_ok=True)
        
        # Copiar imágenes
        for item in split_data:
            src = f"data/processed/images/{item['image']}"
            dst = f"{split_dir}/images/{item['image']}"
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Guardar anotaciones
        with open(f"{split_dir}/labels.json", 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Split '{split_name}' guardado en {split_dir}")
    
    return splits

def show_dataset_stats(dataset):
    """
    Muestra estadísticas del dataset
    """
    print("\n" + "="*60)
    print("ESTADÍSTICAS DEL DATASET")
    print("="*60)
    
    total_words = sum(len(item['words']) for item in dataset)
    total_chars = sum(
        len(word['value']) 
        for item in dataset 
        for word in item['words']
    )
    
    # Contar tipos de texto
    digital_count = 0
    handwritten_count = 0
    
    for item in dataset:
        for word in item['words']:
            if word.get('type', 'digital') == 'digital':
                digital_count += 1
            else:
                handwritten_count += 1
    
    print(f"📊 Total de imágenes: {len(dataset)}")
    print(f"📝 Total de palabras anotadas: {total_words}")
    print(f"🔤 Total de caracteres: {total_chars}")
    print(f"💻 Palabras digitales: {digital_count}")
    print(f"✍️ Palabras manuscritas: {handwritten_count}")
    print(f"📏 Promedio palabras/imagen: {total_words/len(dataset):.1f}")
    print(f"📏 Promedio caracteres/palabra: {total_chars/total_words:.1f}")
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("CONVERSIÓN DE ANOTACIONES A FORMATO docTR")
    print("="*60)
    
    # Rutas
    annotations_file = 'data/annotated/annotations.json'
    images_folder = 'data/raw'
    output_folder = 'data/processed'
    
    # Verificar que existen los archivos
    if not os.path.exists(annotations_file):
        print(f"❌ No se encontró {annotations_file}")
        print("   Asegúrate de haber exportado las anotaciones desde Label Studio")
        exit(1)
    
    if not os.path.exists(images_folder):
        print(f"❌ No se encontró la carpeta {images_folder}")
        exit(1)
    
    # Convertir anotaciones
    dataset = convert_labelstudio_to_doctr(
        annotations_file, 
        images_folder, 
        output_folder
    )
    
    if len(dataset) == 0:
        print("❌ No se pudo crear el dataset. Revisa las anotaciones.")
        exit(1)
    
    # Mostrar estadísticas
    show_dataset_stats(dataset)
    
    # Dividir en train/val/test
    print("\n¿Dividir dataset en train/val/test? (s/n): ", end="")
    should_split = input().lower()
    
    if should_split == 's':
        splits = split_dataset(dataset)
        print("\n✅ Dataset dividido y listo para entrenar!")
    else:
        print("\n💾 Dataset completo guardado en data/processed/")
    
    print("\n🎯 Siguiente paso: Entrenar el modelo")
    print("   Ejecuta: python scripts/04_train_model.py")