from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/change_text', methods=['POST'])
def change_text():
    # Add your server-side logic here
    new_text = 'Text changed using Python!'
    return new_text

if __name__ == '__main__':
    app.run(debug=True)
