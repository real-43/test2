from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)

@app.route('/exec')
def parse(name=None):
    import face_recognize
    print("done")
    return render_template('index.html',name=name)

@app.route('/exec2')
def parse1(name=None):
	import create_data
	print("done")
	return render_template('index.html',name=name)

@app.route('/faceR')
def faceR():
    import face_rec as fr
    return Response(fr.main("test_images/full.jpg"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/faceD')
def faceD():
    import face_detection as fd
    return Response(fd.main(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='1234')