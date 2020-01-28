#run.py
""" Entry point """
from api.app import create_app
from api.config import DevelopmentConfig

application = create_app(DevelopmentConfig)


if __name__ == '__main__':
    application.run()