#!/bin/bash

# Cargar variables de entorno desde .env si existe
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

# Configuración de Gunicorn
PORT=${API_PORT:-5000}
WORKERS=2
BIND_ADDR="0.0.0.0:${PORT}"
APP_ENTRYPOINT="api.app:identity_app"

echo "Iniciando Gunicorn..."
echo "Workers: $WORKERS"
echo "Address: $BIND_ADDR"
echo "App: $APP_ENTRYPOINT"

exec gunicorn --workers $WORKERS \
               --bind $BIND_ADDR \
               --log-level info \
               --access-logfile - \
               $APP_ENTRYPOINT