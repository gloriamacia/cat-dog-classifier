import os

from flask import Flask, render_template, request, redirect
from inference import get_prediction
from fastai2.text.all import *
from commons import *
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_name, class_prob = get_prediction(image_bytes=img_bytes)
        return render_template('result.html', class_prob=class_prob,
                               class_name=class_name)
    return render_template('index.html')


@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def nlp():
    context = dict()

    if request.method == 'POST':
        context['method'] = 'POST'
        model_path = os.getcwd() + '/movie_predictor_model/imdb-sample.pkl'
        learn_inf = load_learner(model_path)
        # context['data'] = movie_sentiment_analysis_predict(request.form['user_input'], learn_inf)
        context['data'] = learn_inf.predict(request.form['user_input'])[0].capitalize()

    return render_template('nlp.html', context=context)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
