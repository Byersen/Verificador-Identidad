import torch
from facenet_pytorch import MTCNN
from PIL import Image
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SOURCE_DIRS = {
    "me": Path("data/me"),
    "not_me": Path("data/not_me")
}
TARGET_ROOT = Path("data/cropped")
IMAGE_SIZE = 160
IMAGE_MARGIN = 20
MIN_FACE_SIZE = 40

def initialize_detector(device):
    """Inicializa el detector de rostros MTCNN."""
    logging.info(f"Usando dispositivo: {device}")
    
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE, 
        margin=IMAGE_MARGIN, 
        min_face_size=MIN_FACE_SIZE, 
        keep_all=False,
        device=device,
        select_largest=True
    )
    return mtcnn

def process_directory(detector, source_dir, target_dir):
    """Procesa todas las imágenes en un directorio fuente y guarda los recortes."""
    if not source_dir.exists():
        logging.warning(f"Directorio fuente no encontrado: {source_dir}. Saltando.")
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(source_dir.glob('*.[jp][pn]g')) + list(source_dir.glob('*.jpeg'))
    logging.info(f"Encontradas {len(image_files)} imágenes en {source_dir}")
    
    processed_count = 0
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            save_path = target_dir / f"{img_path.stem}.png"
            
            detector(img, save_path=str(save_path))
            
            if save_path.exists():
                processed_count += 1
                logging.info(f"Guardado rostro de {img_path} en {save_path}")
            else:
                logging.warning(f"No se detectó rostro en {img_path}")
                
        except Exception as e:
            logging.error(f"Error procesando {img_path}: {e}")
            
    return processed_count

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    face_detector = initialize_detector(device)
    
    total_processed = 0
    for class_label, source_path in SOURCE_DIRS.items():
        target_path = TARGET_ROOT / class_label
        logging.info(f"--- Procesando clase: {class_label} ---")
        count = process_directory(face_detector, source_path, target_path)
        total_processed += count
        
    logging.info(f"--- Proceso completado ---")
    logging.info(f"Total de rostros recortados y guardados: {total_processed}")

if __name__ == "__main__":
    main()
