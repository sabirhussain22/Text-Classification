from flask import Flask, render_template, request
import nltk
import pickle
import joblib
from nltk.util import pr
from nltk.tokenize import word_tokenize      #to divide strings into lists of substrings
from nltk.stem import WordNetLemmatizer      #to link words with similar meanings to one word.
from nltk.corpus import stopwords            #to filterout useless data
stopword = set(stopwords.words('english'))
import re
import string

def clean(text):
    text = str(text).lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub(r"\@w+|\#",'',text)
    text = re.sub(r"[^\w\s]",'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    tweet_tokens = word_tokenize(text)
    filtered_tweets = [w for w in tweet_tokens if not w in stopword]  # assuming stopword is defined somewhere
    return " ".join(filtered_tweets)



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        text_input = request.form['text_input']
        #applying pre-processing to text data
        tweet = clean(text_input)  # Clean the text
        lemmatizer = WordNetLemmatizer()
        tweet = lemmatizer.lemmatize(tweet)

        print(tweet)
        hate_model = joblib.load('Trained Models\\logreg_hate.joblib')
        nor_model= joblib.load('Trained Models\\ln.joblib')
        vect = joblib.load('Trained Models\\TfidfVectorizer.joblib')

        X = vect.transform([tweet])  # Wrap the tweet in a list
        if hate_model.predict(X)=='Hate Speech':
            result='Hate Speech'
        elif nor_model.predict(X)=='Normal':
            result='Normal'
        else:
            result='Offensive'
        # Perform your classification logic here
        result = "Classification result: " + result
        return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
