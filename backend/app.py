from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange
import numpy as np 
from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def return_prediction(model, sample_json):
 
    max_features = 2500
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    
    headline = sample_json['comment']
    headline = tokenizer.texts_to_sequences(headline)
    headline = pad_sequences(headline, maxlen=78, dtype='int32', value=0)

    sentiment = model.predict(headline,batch_size=1,verbose = 2)[0]

    # threshold = 0.6
    # if sentiment[1] > threshold:
    #   return "Sarcasm"
    # else:
    #    return "Non-Sarcastic"

    if(np.argmax(sentiment) == 0):
        return "Non-Sarcastic"
    elif (np.argmax(sentiment) == 1):
        return "Sarcasm"

app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'SECRET_KEY'
# Loading the model and scaler
model = load_model('final_model2.h5')


class CommentForm(FlaskForm):
    comment = TextField('Social Media Comment')
    submit = SubmitField('Analyze')
 
@app.route('/', methods=['GET', 'POST'])
def index():
    # Create instance of the form.
    form = CommentForm()
    # If the form is valid on submission
    if form.validate_on_submit():
    # Grab the data from the input on the form.
        session['comment'] = form.comment.data
        return redirect(url_for('prediction'))
    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['comment'] = str(session['comment'])
    

    results = return_prediction(model=model, sample_json=content)

    return render_template('prediction.html',results=results)

if __name__ == '__main__':
 app.run(debug=True)