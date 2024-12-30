import os
from dotenv import load_dotenv

load_dotenv()

def load_config(app):
    app.config['ENV'] = os.getenv('FLASK_ENV', 'production')
    app.config['DEBUG'] = bool(int(os.getenv('FLASK_DEBUG', '0')))
    app.config['PORT'] = int(os.getenv('PORT', '5050'))