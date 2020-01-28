from flask import Flask

from api.config import get_logger


_logger = get_logger(logger_name=__name__)


def create_app(config_object):
    """creates instance of flask application"""

    flask_app = Flask('ml_api')
    flask_app.config.from_object(config_object)

    from api.controller import prediction_app
    flask_app.register_blueprint(prediction_app)
    _logger.debug('flask application instance created')

    return flask_app