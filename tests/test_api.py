import pytest
import requests
import os

BASE_URL = "http://127.0.0.1:5000" 
SAMPLES_DIR = "samples"


def find_sample_path(preferred_name, fallback_dir):
    """Busca preferred_name dentro de samples/ y si no existe, toma el primer jpg/png en fallback_dir."""
    pref = os.path.join(SAMPLES_DIR, preferred_name)
    if os.path.exists(pref):
        return pref
    if os.path.isdir(fallback_dir):
        for ext in ("jpg", "jpeg", "png"):
            files = [f for f in os.listdir(fallback_dir) if f.lower().endswith(f".{ext}")]
            if files:
                return os.path.join(fallback_dir, files[0])
    return pref


ME_IMAGE_PATH = find_sample_path("selfie.jpg", os.path.join("data", "me"))
NOT_ME_IMAGE_PATH = find_sample_path("other.jpg", os.path.join("data", "not_me"))
INVALID_FILE_PATH = os.path.join(SAMPLES_DIR, "test.txt")

# Fixture para ver si la api esta viva
@pytest.fixture(scope="module")
def api_is_running():
    try:
        response = requests.get(f"{BASE_URL}/healthz", timeout=2)
        assert response.status_code == 200
    except requests.ConnectionError:
        pytest.fail(f"No se pudo conectar a la API en {BASE_URL}. ¿Está corriendo?")

def test_health_check(api_is_running):
    """Prueba el endpoint /healthz."""
    response = requests.get(f"{BASE_URL}/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # CORRECCIÓN: La API devuelve 'version', no 'model_version'
    assert "version" in data

@pytest.mark.skipif(not os.path.exists(ME_IMAGE_PATH), reason="Archivo de prueba 'me' no encontrado")
def test_verify_is_me(api_is_running):
    """Prueba una imagen positiva ('YO')."""
    with open(ME_IMAGE_PATH, 'rb') as f:
        files = {'image': (os.path.basename(ME_IMAGE_PATH), f, 'image/jpeg')}
        response = requests.post(f"{BASE_URL}/verify", files=files)
        
    assert response.status_code == 200
    data = response.json()
    assert data["is_me"] == True
    assert "score" in data
    assert data["score"] > data["threshold"]

@pytest.mark.skipif(not os.path.exists(NOT_ME_IMAGE_PATH), reason="Archivo de prueba 'not_me' no encontrado")
def test_verify_is_not_me(api_is_running):
    """Prueba una imagen negativa ('NO-YO')."""
    with open(NOT_ME_IMAGE_PATH, 'rb') as f:
        files = {'image': (os.path.basename(NOT_ME_IMAGE_PATH), f, 'image/jpeg')}
        response = requests.post(f"{BASE_URL}/verify", files=files)
        
    assert response.status_code == 200
    data = response.json()
    assert data["is_me"] == False
    assert "score" in data

def test_verify_no_file(api_is_running):
    """Prueba una solicitud sin el archivo 'image'."""
    response = requests.post(f"{BASE_URL}/verify")
    assert response.status_code == 400
    assert "Falta el archivo 'image'" in response.json()["error"]

@pytest.mark.skipif(not os.path.exists(INVALID_FILE_PATH), reason="Archivo de prueba 'invalid' no encontrado")
def test_verify_invalid_mimetype(api_is_running):
    """Prueba un tipo de archivo no permitido (ej. text/plain)."""
    if not os.path.exists(INVALID_FILE_PATH):
        try:
            os.makedirs(SAMPLES_DIR, exist_ok=True)
            with open(INVALID_FILE_PATH, 'w') as f:
                f.write("this is a test file")
        except OSError:
             pytest.skip("No se pudo crear el archivo de prueba 'invalid' y no existe.")

    with open(INVALID_FILE_PATH, 'rb') as f:
        files = {'image': (os.path.basename(INVALID_FILE_PATH), f, 'text/plain')}
        response = requests.post(f"{BASE_URL}/verify", files=files)
        
    assert response.status_code == 415
    assert "solo image/jpeg o image/png" in response.json()["error"]
