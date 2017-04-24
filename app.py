APP_NAME = 'GREP(v1.0.0) RESTful API'
BASE_URL = '127.0.0.1:5000'
APP_DESCRIPTION = 'Happiness intensity estimation of a group image'


import time
from flask import Flask, jsonify, g, request, redirect
from flask_swagger import swagger

app = Flask(__name__)

@app.route('/')
def root():
    print('root activated')
    return redirect('./static/demo/index.html')

@app.route('/doc')
def doc():
    print('doc activated')
    return redirect('./static/doc/index.html')
# @app.route('/static/demo/img/brain.svg')
# def serve_content(svgFile):
#     return file('static/'+svgFile+'.svg').read()

@app.route('/spec')
def spec():
    print('spec activated')
    swag = swagger(app)
    swag['info']['version'] = "v1.0.0"
    swag['info']['title'] = APP_NAME
    swag['info']['description'] = APP_DESCRIPTION
    swag['info']['host'] = BASE_URL
    return jsonify(swag)


@app.before_request
def before_request():
    g.start_time = time.time()


@app.after_request
def after_request(response):
    diff = time.time() - g.start_time
    diff *= 1000.0  # to convert from seconds to milliseconds
    info = '%.4f ms <-- %s' % (diff, request.path)
    app.logger.info(info)
    return response


import error
import admin
import GREP_v100
