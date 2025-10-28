"""
Script completo para entrenar modelo OCR con docTR
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from doctr.models import recognition
from doctr import transforms as T
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import datetime

# ==================== DATASET PERSONALIZADO ====================

class CustomOCRDataset(Dataset):
    """Dataset personalizado para docTR"""
    
    def __init__(self, labels_path, images_folder, vocab=None, img_transforms=None):
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.images_folder = images_folder
        self.img_transforms = img_transforms
        
        # Crear vocabulario si no existe
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
    
    def _build_vocab(self):
        """Construye vocabulario a partir de los datos"""
        chars = set()
        for item in self.data:
            for word in item['words']:
                chars.update(word['value'])
        
        # Agregar caracteres especiales
        vocab = ['<pad>', '<sos>', '<eos>'] + sorted(list(chars))
        print(f"📚 Vocabulario construido: {len(vocab)} caracteres únicos")
        return vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Cargar imagen
        img_path = os.path.join(self.images_folder, item['image'])
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.img_transforms:
            image = self.img_transforms(image)
        else:
            # Transformación por defecto
            image = T.Resize((32, 128))(image)
            image = np.array(image).transpose(2, 0, 1) / 255.0
            image = torch.FloatTensor(image)
        
        # Concatenar todo el texto de la imagen
        full_text = ' '.join([word['value'] for word in item['words']])
        
        # Codificar texto
        target = self._encode_text(full_text)
        
        return image, target, full_text
    
    def _encode_text(self, text):
        """Convierte texto a secuencia de índices"""
        return [self.char_to_idx.get(char, 0) for char in text]

# ==================== FUNCIONES DE ENTRENAMIENTO ====================

def collate_fn(batch):
    """Función para agrupar batch con longitudes variables"""
    images, targets, texts = zip(*batch)
    
    # Stack de imágenes
    images = torch.stack(images, 0)
    
    # Pad de targets
    max_len = max([len(t) for t in targets])
    padded_targets = []
    target_lengths = []
    
    for target in targets:
        target_lengths.append(len(target))
        padded = target + [0] * (max_len - len(target))
        padded_targets.append(padded)
    
    padded_targets = torch.LongTensor(padded_targets)
    target_lengths = torch.LongTensor(target_lengths)
    
    return images, padded_targets, target_lengths, texts

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Entrena un epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets, target_lengths, texts) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calcular loss (simplificado - ajustar según modelo exacto)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Actualizar barra de progreso
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Valida el modelo"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets, target_lengths, texts in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Guarda checkpoint del modelo"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"💾 Checkpoint guardado: {filepath}")

# ==================== ENTRENAMIENTO PRINCIPAL ====================

def train_model(config):
    """
    Función principal de entrenamiento
    
    config: diccionario con configuración
    """
    print("="*70)
    print("INICIANDO ENTRENAMIENTO DEL MODELO OCR")
    print("="*70)
    
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Dispositivo: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Crear datasets
    print("\n📚 Cargando datasets...")
    
    # Transformaciones para entrenamiento
    train_transforms = T.Compose([
        T.Resize((32, 128)),
        T.ColorInversion(min_val=0.6),
        T.RandomBrightness(max_delta=0.3),
        T.RandomContrast(delta=0.2),
    ])
    
    # Dataset de entrenamiento
    train_dataset = CustomOCRDataset(
        labels_path=config['train_labels'],
        images_folder=config['train_images'],
        img_transforms=train_transforms
    )
    
    # Dataset de validación
    val_dataset = CustomOCRDataset(
        labels_path=config['val_labels'],
        images_folder=config['val_images'],
        vocab=train_dataset.vocab,  # Usar mismo vocabulario
        img_transforms=T.Compose([T.Resize((32, 128))])
    )
    
    print(f"   Train: {len(train_dataset)} muestras")
    print(f"   Val: {len(val_dataset)} muestras")
    print(f"   Vocabulario: {len(train_dataset.vocab)} caracteres")
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    # Cargar modelo
    print("\n🤖 Cargando modelo...")
    model = recognition.crnn_vgg16_bn(
        pretrained=True,
        vocab=train_dataset.vocab
    )
    model = model.to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parámetros totales: {total_params:,}")
    print(f"   Parámetros entrenables: {trainable_params:,}")
    
    # Configurar optimizador
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Criterio de pérdida
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Directorio para checkpoints
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Variables para tracking
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Log de entrenamiento
    log_file = f"{config['checkpoint_dir']}/training_log.txt"
    
    print("\n🚀 Iniciando entrenamiento...")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Early stopping patience: {config['early_stopping_patience']}")
    print("="*70)
    
    # Loop de entrenamiento
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n📊 Epoch {epoch}/{config['num_epochs']}")
        print("-" * 70)
        
        # Entrenar
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validar
        val_loss = validate(model, val_loader, criterion, device)
        
        # Actualizar scheduler
        scheduler.step(val_loss)
        
        # Obtener learning rate actual
        current_lr = optimizer.param_groups[0]['lr']
        
        # Imprimir resultados
        print(f"\n📈 Resultados Epoch {epoch}:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Learning Rate: {current_lr:.6f}")
        
        # Guardar log
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}\n")
        
        # Guardar checkpoint cada N epochs
        if epoch % config['save_every'] == 0:
            checkpoint_path = f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = f"{config['checkpoint_dir']}/best_model.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            print(f"✨ ¡Nuevo mejor modelo! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{config['early_stopping_patience']}")
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n🛑 Early stopping activado en epoch {epoch}")
            break
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"📊 Mejor Val Loss: {best_val_loss:.4f}")
    print(f"💾 Modelo guardado en: {config['checkpoint_dir']}/best_model.pth")
    
    # Guardar vocabulario
    vocab_path = f"{config['checkpoint_dir']}/vocab.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(train_dataset.vocab, f, ensure_ascii=False, indent=2)
    print(f"📚 Vocabulario guardado en: {vocab_path}")
    
    return model, train_dataset.vocab

# ==================== CONFIGURACIÓN Y EJECUCIÓN ====================

if __name__ == "__main__":
    # Configuración de entrenamiento
    config = {
        # Datos
        'train_labels': 'data/splits/train/labels.json',
        'train_images': 'data/splits/train/images',
        'val_labels': 'data/splits/val/labels.json',
        'val_images': 'data/splits/val/images',
        
        # Hiperparámetros
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        
        # Training
        'num_workers': 4,
        'save_every': 5,
        'early_stopping_patience': 10,
        
        # Output
        'checkpoint_dir': 'models/checkpoints',
    }
    
    # Verificar que existen los datos
    required_files = [
        config['train_labels'],
        config['val_labels']
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Faltan archivos necesarios:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 Asegúrate de haber ejecutado:")
        print("   1. python scripts/02_prepare_images.py")
        print("   2. Etiquetar en Label Studio")
        print("   3. python scripts/03_convert_annotations.py")
        exit(1)
    
    # Mostrar configuración
    print("\n⚙️  CONFIGURACIÓN DE ENTRENAMIENTO")
    print("="*70)
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("="*70)
    
    print("\n¿Continuar con el entrenamiento? (s/n): ", end="")
    confirm = input().lower()
    
    if confirm == 's':
        # Entrenar modelo
        model, vocab = train_model(config)
        print("\n🎉 ¡Entrenamiento finalizado exitosamente!")
    else:
        print("❌ Entrenamiento cancelado")