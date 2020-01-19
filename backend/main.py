from flask import Flask, render_template, Response, redirect
import cv2, json
from flask_socketio import send, emit

app = Flask(__name__)
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
    print(suggestion)
    ret = True
    cap = cv2.VideoCapture(0)
    while ret == True:
        ret, frame = cap.read()
        if ret == False:
            break
        suggestion = process_frame(frame)
        # if suggestion != "":
        #     cap.release()
        #     return redirect('/')
        succ, frame = cv2.imencode(".jpg", frame)
        suggestion = ""
        # print(suggestion)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

         


if __name__ == '__main__':
    app.run(debug = True)

 