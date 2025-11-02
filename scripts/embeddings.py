import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import os
from pathlib import Path
import logging
from torchvision import transforms
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

CROPPED_DIRS = {
    "me": (Path("data/cropped/me"), 1),
    "not_me": (Path("data/cropped/not_me"), 0)
}
EMBEDDINGS_FILE = Path(os.getenv("EMBEDDINGS_DATA_PATH", "data/embeddings.npy"))
LABELS_FILE = Path(os.getenv("LABELS_DATA_PATH", "data/labels.csv"))
BATCH_SIZE = 32

def initialize_models(device):
    """Inicializa el generador de embeddings InceptionResnetV1."""
    logging.info(f"Usando dispositivo: {device}")
    embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return embedder

def process_images_in_batches(embedder, device, image_paths):
    """Genera embeddings para una lista de imágenes en lotes (batches)."""
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    all_embeddings = []
    valid_paths = []
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_tensors = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = preprocess(img)
                batch_tensors.append(tensor)
                valid_paths.append(img_path)
            except Exception as e:
                logging.warning(f"No se pudo cargar {img_path}: {e}")

        if not batch_tensors:
            continue
            
        batch = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            embeddings = embedder(batch)
            
        all_embeddings.extend(embeddings.cpu().numpy())
        logging.info(f"Procesado lote {i//BATCH_SIZE + 1}/{len(image_paths)//BATCH_SIZE + 1}")

    return all_embeddings, len(valid_paths)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    embedding_generator = initialize_models(device)
    
    feature_vectors = []
    label_list = []
    
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    for class_label, (source_dir, label) in CROPPED_DIRS.items():
        logging.info(f"--- Generando embeddings para: {class_label} ---")
        
        if not source_dir.exists():
            logging.warning(f"Directorio no encontrado: {source_dir}. Saltando.")
            continue
            
        image_files = list(source_dir.glob('*.png'))
        
        if not image_files:
            logging.warning(f"No se encontraron imágenes en {source_dir}.")
            continue
            
        logging.info(f"Encontradas {len(image_files)} imágenes.")
        
        embeddings, valid_count = process_images_in_batches(embedding_generator, device, image_files)
        
        feature_vectors.extend(embeddings)
        label_list.extend([label] * valid_count)
        
        logging.info(f"Generados {len(embeddings)} embeddings para la clase {class_label}.")

    if not feature_vectors:
        logging.error("No se generaron embeddings. Abortando.")
        return

    X = np.array(feature_vectors)
    y = np.array(label_list)
    
    try:
        np.save(EMBEDDINGS_FILE, X)
        logging.info(f"Embeddings guardados en: {EMBEDDINGS_FILE} (Shape: {X.shape})")
        
        np.savetxt(LABELS_FILE, y, delimiter=',', fmt='%d')
        logging.info(f"Etiquetas guardadas en: {LABELS_FILE} (Shape: {y.shape})")
        
    except Exception as e:
        logging.error(f"Error al guardar archivos: {e}")

if __name__ == "__main__":
    main()