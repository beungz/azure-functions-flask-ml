import logging

import azure.functions as func


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    from FlaskApp import flask_app
    return func.WsgiMiddleware(flask_app.wsgi_app).handle(req, context)
