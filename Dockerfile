# Imagen base de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY ./server /app/server
COPY ./server/requirements.txt /app/requirements.txt

# Instalar dependencias
RUN pip install --no-cache-dir -r /app/requirements.txt

# Verificar las dependencias instaladas (para debug)
RUN pip freeze

# Exponer el puerto en el contenedor
EXPOSE 5010

# Comando para ejecutar la aplicación
CMD ["python", "/app/server/run.py"]