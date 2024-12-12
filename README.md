# Modelo-T5

Este proyecto utiliza el modelo **T5 (Text-to-Text Transfer Transformer)** desarrollado por Google, implementado en un servidor Flask para realizar tareas como:

- Traducción de texto.
- Resumen de texto.
- Generación de texto.

## Características

- **Dockerizado**: Puedes desplegar el proyecto fácilmente usando Docker.
- **Endpoints de API**:
  - `/api/t5/translate`: Traducción entre idiomas.
  - `/api/t5/summarize`: Resumen de texto.
  - `/api/t5/generate`: Generación de texto basado en un prompt.

## Requisitos

- Python 3.10+
- Docker (opcional, para despliegue rápido)

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Aurosonic7/Modelo-T5.git
   cd Modelo-T5