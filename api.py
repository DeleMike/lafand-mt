'''
Flask Application
'''
from flask import Flask, jsonify, request


from diacritize_t5_small import (diacritize)

app = Flask(__name__)

@app.route('/diacritize', methods=['GET', 'POST'])
def hello_world():
    '''
    Returns a JSON test message
    '''
    output = ''
    if request.method == 'POST':
        body = request.json
        input_val = str(body['text'])
        output = diacritize(text=input_val)
        
    return jsonify({"result": output})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)