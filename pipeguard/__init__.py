from flask import Flask
from .config import Config
from .db import Database
from .engine import LiveEngine


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.db = Database(app.config["DATABASE"])
    app.engine = LiveEngine(app.db, config_class)
    from .routes import bp
    app.register_blueprint(bp)
    if app.config.get("START_ENGINE"):
        app.engine.start()
    return app
