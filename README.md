🛡️ Verificador de identidad por imagen

Verificador binario de identidad (YO vs NO-YO) basado en embeddings faciales y clasificador ligero (LogisticRegression).

🎯 Objetivo

El proyecto entrena un clasificador simple para responder a la pregunta: "¿Es la persona en esta imagen YO?". Utiliza el Deep Learning preentrenado (FaceNet) para generar vectores numéricos (embeddings) y Machine Learning clásico (LogisticRegression) para la clasificación final.

🚀 1. Configuración Inicial y Dependencias

Sigue esta secuencia de comandos desde tu terminal (ajustando la activación del entorno virtual según uses Linux/Mac o Windows) para iniciar.

    Clonar el Repositorio
    Bash

git clone <repository-url>
cd repositorio

Crear y Activar Entorno Virtual
Bash

python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
.\venv\Scripts\activate

Instalar Dependencias
Bash

pip install -r requirements.txt

(Nota: Si usas GPU, instala la rueda de PyTorch manualmente antes de este paso).

Configurar Variables de Entorno
Bash

    cp .env.example .env
    # Edita el archivo .env con tus configuraciones si es necesario.

🏃‍♂️ 2. Flujo de Trabajo y Entrenamiento

El pipeline de generación del modelo debe seguir estos cuatro pasos en orden:

Paso 0: Preparación de Datos

Antes de ejecutar los comandos, debes colocar tus fotos en las carpetas:

    data/me/: Mínimo 50-100 fotos variadas de tu persona.

    data/not_me/: Varias fotos de otras personas (200+ recomendado).

Paso 1: Recortar Rostros

Detecta rostros con MTCNN y los recorta a 160x160px.
Bash

python scripts/crop_faces.py

Paso 2: Generar Embeddings

Extrae el vector de características (512D) con InceptionResnetV1.
Bash

python scripts/embeddings.py

Paso 3: Entrenar Clasificador

Entrena el modelo final y guarda los artefactos (models/model.joblib).
Bash

python train.py

Paso 4: Evaluar Modelo

Calcula el rendimiento, genera gráficos y determina el umbral óptimo de decisión (best_threshold).
Bash

python evaluate.py

🌐 3. Ejecución y Prueba de la API

Una vez generados los artefactos en la carpeta models/, puedes levantar el servidor para probar la verificación.

Ejecutar API Localmente

Bash

# Modo desarrollo
python api/app.py

# Modo producción (Usando Gunicorn)
./scripts/run_gunicorn.sh

Probar la API

Puedes usar curl para verificar los endpoints.

    Verificar Estado del Servicio (/healthz):
    Bash

curl http://localhost:5000/healthz
# Respuesta esperada: {"status": "ok", "best_threshold": 0.72}

Verificar Identidad (/verify):
Bash

    # Asegúrate de cambiar "path/to/image.jpg" por la ubicación real de una foto.
    curl -X POST \
         -F "image=@path/to/image.jpg" \
         http://localhost:5000/verify
    # Respuesta esperada: {"is_me": true, "score": 0.8421, "threshold": 0.75}

📁 4. Estructura y Artefactos

El proyecto organiza sus archivos por función y etapa del pipeline:

    api/: Contiene el servidor Flask (app.py).

    scripts/: Contiene utilidades como crop_faces.py y run_gunicorn.sh.

    data/: Es la zona de entrada/salida. Aquí están las fotos crudas, los recortes y los embeddings.npy.

    models/: Contiene los artefactos entrenados (model.joblib y scaler.joblib).

    reports/: Contiene las métricas y los gráficos PNG del proceso de evaluate.py.

    ⚠️ Importante sobre Privacidad: Las carpetas data/ y models/ contienen información sensible y son grandes. Debes asegurarte de que estén en tu archivo .gitignore y nunca subirlas al repositorio.

🔒 5. Consideraciones de Seguridad

Los datos de entrenamiento son sensibles. En entornos de producción, el endpoint /verify debe estar protegido por autenticación (ej. JWT) y todo el tráfico debe ser cifrado con HTTPS. Evita exponer la API sin restricciones.