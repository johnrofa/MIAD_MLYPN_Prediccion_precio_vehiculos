#!/usr/bin/python
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import joblib
from m09_model_deployment_01 import predict_proba, Modelo
import json

app = Flask(__name__)


api = Api(
    app,
    version='1.0',
    title='Prediccion Precio Vehiculos',
    description='Prediccion Precio Vehiculos')

ns = api.namespace('predict',
                   description='Prediccion Precio Vehiculos')

parser = api.parser()

parser.add_argument(
    'URL',
    type=str,
    required=True,
    help='URL to be analyzed',
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})


@app.route('/modelo1', methods=["POST"])
def lectura():
    print(request.data)
    datos1 = json.loads(request.data)
    print('datos1: ', datos1)
    resultado= Modelo(datos1)
    costovehiculo = {
        "El costo del vehiculo es ": resultado,
    }

    return jsonify(costovehiculo), 200



@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()

        return {
                   "result": predict_proba(args['URL'])
               }, 200


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
