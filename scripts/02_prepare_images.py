"""
Script para preparar y organizar imágenes para etiquetar
"""
import os
import shutil
from pathlib import Path
from PIL import Image
import random

def prepare_images_for_labeling(source_folder, output_folder, sample_size=None):
    """
    Prepara imágenes para Label Studio
    
    Args:
        source_folder: Carpeta con tus imágenes originales
        output_folder: Carpeta de salida (data/raw)
        sample_size: Número de imágenes a tomar (None = todas)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Extensiones válidas
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    
    # Listar todas las imágenes
    all_images = []
    for ext in valid_extensions:
        all_images.extend(Path(source_folder).glob(f"**/*{ext}"))
        all_images.extend(Path(source_folder).glob(f"**/*{ext.upper()}"))
    
    print(f"📁 Encontradas {len(all_images)} imágenes")
    
    # Tomar muestra si se especifica
    if sample_size and sample_size < len(all_images):
        all_images = random.sample(all_images, sample_size)
        print(f"📊 Tomando muestra de {sample_size} imágenes")
    
    # Copiar y renombrar imágenes
    copied = 0
    for idx, img_path in enumerate(all_images):
        try:
            # Verificar que la imagen se puede abrir
            img = Image.open(img_path)
            img.verify()
            
            # Nuevo nombre
            new_name = f"image_{idx:05d}{img_path.suffix.lower()}"
            new_path = os.path.join(output_folder, new_name)
            
            # Copiar imagen
            shutil.copy2(img_path, new_path)
            copied += 1
            
            if (idx + 1) % 100 == 0:
                print(f"  Procesadas {idx + 1} imágenes...")
                
        except Exception as e:
            print(f"❌ Error con {img_path}: {e}")
    
    print(f"\n✅ {copied} imágenes preparadas en {output_folder}")
    return copied

def generate_synthetic_images(output_folder, num_images=100):
    """
    Genera imágenes sintéticas para entrenamiento
    """
    from PIL import ImageDraw, ImageFont
    import random
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Textos de ejemplo (mezcla de digital y manuscrito simulado)
    sample_texts = [
        "Factura N° 001-234",
        "TOTAL: S/ 150.00",
        "Nombre: Juan Pérez",
        "Fecha: 27/10/2025",
        "Producto: Laptop HP",
        "Cantidad: 2 unidades",
        "Dirección: Av. Larco 123",
        "RUC: 20123456789",
        "Teléfono: 987654321",
    ]
    
    fonts_to_try = [
        "arial.ttf", "times.ttf", "courier.ttf", "verdana.ttf",
        "calibri.ttf", "georgia.ttf", "impact.ttf"
    ]
    
    print(f"🎨 Generando {num_images} imágenes sintéticas...")
    
    for i in range(num_images):
        # Crear imagen
        width, height = random.randint(400, 800), random.randint(100, 300)
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Seleccionar texto y fuente
        text = random.choice(sample_texts)
        font_size = random.randint(20, 48)
        
        try:
            font_name = random.choice(fonts_to_try)
            font = ImageFont.truetype(font_name, font_size)
        except:
            font = ImageFont.load_default()
        
        # Posición aleatoria
        x = random.randint(10, 50)
        y = random.randint(10, height - 50)
        
        # Color de texto aleatorio (oscuro)
        color = (
            random.randint(0, 50),
            random.randint(0, 50),
            random.randint(0, 50)
        )
        
        # Dibujar texto
        draw.text((x, y), text, fill=color, font=font)
        
        # Guardar
        img.save(f"{output_folder}/synthetic_{i:05d}.png")
        
        if (i + 1) % 50 == 0:
            print(f"  Generadas {i + 1} imágenes...")
    
    print(f"✅ Imágenes sintéticas guardadas en {output_folder}")

if __name__ == "__main__":
    print("=" * 60)
    print("PREPARACIÓN DE DATOS PARA ENTRENAMIENTO")
    print("=" * 60)
    
    # Opción 1: Si tienes imágenes propias
    print("\n1️⃣ ¿Tienes imágenes propias? (s/n): ", end="")
    has_images = input().lower()
    
    if has_images == 's':
        print("📂 Ruta de la carpeta con tus imágenes: ", end="")
        source = input()
        if os.path.exists(source):
            prepare_images_for_labeling(source, 'data/raw')
        else:
            print("❌ Carpeta no encontrada")
    
    # Opción 2: Generar imágenes sintéticas
    print("\n2️⃣ ¿Generar imágenes sintéticas? (s/n): ", end="")
    gen_synthetic = input().lower()
    
    if gen_synthetic == 's':
        print("🔢 ¿Cuántas imágenes generar? (default: 100): ", end="")
        num = input()
        num = int(num) if num.isdigit() else 100
        generate_synthetic_images('data/raw/synthetic', num)
    
    print("\n✅ Preparación completada!")
    print("📁 Tus imágenes están en: data/raw/")
    print("🏷️ Siguiente paso: Etiquetar con Label Studio")