#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : b站@在下啊水
from flask import Flask, render_template, Response, request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import sys 
sys.path.append('D:\Project\positionDetect\yolov5-master\yolov5-master\\') 
from detect import *
import os
import torch
from importlib import import_module
# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from yolov5_flask import Camera
from flask_cors import *

app = Flask(__name__)
# UPLOAD_FOLDER = "C:\Users\Arpit Sharma\Desktop\Friendship goals\content\yolov5\static\uploads"
DETECTION_FOLDER = r'./static/detections'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
#app.config['DETECTION_FOLDER'] = DETECTION_FOLDER
import datetime
import json

def crossDomainResponse(data):
    response = Response(data)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response

@app.route('/log')
def log():
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return crossDomainResponse(json.dumps({'out':now_time+' '+Camera.res}, ensure_ascii=False))


@app.route('/index')
def index():
    """Video streaming home page."""
    return render_template('page.html')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        # a = camera.people_appeal()
        # print('a:{}0'.format(a))
        # for i in a:
        #     if i =='people':
        #         print('是people：{}}')
        #         people_appeal()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')





if __name__ == "__main__":
    app.run(debug = True)
