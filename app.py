import sys
sys.path.append('.')

import os
import numpy as np
import base64
import io
from flask_cors import CORS

from flask import Flask, request, jsonify
import cv2
from infer import img_infer

app = Flask(__name__) 
CORS(app)

@app.route('/alpr', methods=['POST'])
def alpr_process():
    alpr_results = []
    file = request.files['file'].read()

    try:
        file_bytes = np.frombuffer(file, np.uint8)
        _ = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    except:
        response = jsonify({"alprs": alpr_results})

        response.status_code = 200
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response


    recog_results, recog_boxes, vehicle_boxes = img_infer(file)

    for i in range(len(recog_results)) :
        alpr_results.append({"recog_results": recog_results[i], "recog_boxes": recog_boxes[i], "vehicle_boxes": vehicle_boxes[i][:]})
        
    response = jsonify({"alprs": alpr_results})

    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8083))
    app.run(host='0.0.0.0', port=port)