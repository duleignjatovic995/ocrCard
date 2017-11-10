from flask import request
from flask import Flask
import extractor
import os
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def parse_request():
    if request.method == 'GET':
        return "Use POST method!"
    if request.method == 'POST' and 'image' in request.files:
        filename = "{}.png".format(os.getpid())
        request.files['image'].save(filename)
        response = extractor.extract_image(filename)
        return response

if __name__ == '__main__':
    app.run()
