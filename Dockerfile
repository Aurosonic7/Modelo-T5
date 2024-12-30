# Dockerfile
FROM python:3.11-slim

# Establecer el directorio de trabajo dentro del contenedor a /app/server
WORKDIR /app/server

# Copiar todos los archivos necesarios al contenedor
COPY ./server-flask /app/server

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# (Opcional) Verificar las dependencias instaladas para debugging
RUN pip freeze

# Exponer el puerto en el contenedor
EXPOSE 5050

# Comando para ejecutar la aplicación usando Gunicorn
CMD ["gunicorn", "-c", "gunicorn_config.py", "wsgi:app"]