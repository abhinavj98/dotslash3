from flask import Flask, render_template, Response, redirect
import cv2, json
from flask_socketio import send, emit


app = Flask(__name__, static_folder='static')
suggestion = ""
def process_frame(frame):
    return "nice"
    
@app.route('/suggestion')
def sugg():
    print("a")
    global suggestion
    return (suggestion)

@app.route('/')
def index():
    # return "Hello World!"
    return render_template('index.html', my_func=sugg)

def gen():
    global suggestion
    ret = True
    cap = cv2.VideoCapture(0)
    while ret == True:
        ret, frame = cap.read()
        if ret == False:
            break
        suggestion = process_frame(frame)
        succ, frame = cv2.imencode(".jpg", frame)
        suggestion = ""
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def example_gen(path):
    ret = True
    cap = cv2.VideoCapture(path)
    while ret == True:
        ret, frame = cap.read()
        if ret == False:
            break
        suggestion = process_frame(frame)
        succ, frame = cv2.imencode(".jpg", frame)
        suggestion = ""
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')
    cap.release()


@app.route('/example_feed/<path>')
def video_example(path):
    return Response(example_gen(path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
         


@app.route('/e1')
def e1():
    return render_template('e1.html')

@app.route('/e2')
def e2():
    return render_template('e2.html')

@app.route('/e3')
def e3():
    return render_template('e3.html')

if __name__ == '__main__':
    app.run(debug = True)

 