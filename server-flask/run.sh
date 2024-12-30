#!/usr/bin/env bash
source .env
exec gunicorn -c gunicorn_config.py wsgi:app