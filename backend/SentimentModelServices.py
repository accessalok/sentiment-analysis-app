from flask import Flask, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.datasets import imdb
max_length = 256  # Maximum length of the sequences
padding_type = 'post'  # Padding type for sequences shorter than the maximum length
# Replace with your actual word_index dictionary
word_index = word_index = imdb.get_word_index()

# Download VADER lexicon
nltk.download('vader_lexicon')
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
keras_model = load_model('model/sentiment_model.keras')


# Initialize the app and VADER sentiment analyzer
app = Flask(__name__)
sia = SentimentIntensityAnalyzer()
# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

def cleanup_text(text):
    # Convert to lower case
    text = text.lower()
    # Remove hashtags
    text = re.sub(r'\B#\S+', '', text)
    # Remove links
    text = re.sub(r"http\S+", "", text)
    # Remove special characters
    text = ' '.join(re.findall(r'\w+', text))
    # Substitute multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', '', text)
    # Remove Twitter handles
    text = re.sub('@[^\s]+', '', text)
    
    return text

def preprocess_text(text):
    # Tokenization
    tokenized_words = nltk.word_tokenize(text)

    # Initialize lemmatizer, stemmer, and stop words
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Stemming, Lemmatization, and Stop Words Removal
    processed_words = []
    for word in tokenized_words:
        # Convert to lower case
        lower_word = word.lower()
        # Skip stop words
        if lower_word in stop_words:
            continue
        # Lemmatization
        lemmatized_word = lemmatizer.lemmatize(lower_word)
        # Stemming
        #stemmed_word = stemmer.stem(lemmatized_word)
        processed_words.append(lemmatized_word)

    return processed_words

    
def get_sentiwordnet_scores(text):
    # Preprocess the text
    processed_text = cleanup_text(text)
    processed_words = preprocess_text(processed_text)
    print(processed_words)
    
    # POS tagging
    posTagSentence = nltk.pos_tag(processed_words)
    
    sentiment_scores = {
        'pos': 0.0,
        'neg': 0.0,
        'neu': 0.0,
        'compound': 0.0
    }

    for wordTagPair in posTagSentence:
        word = wordTagPair[0]
        posTag = wordTagPair[1]

        if posTag.startswith('J'):
            posTag = wn.ADJ
        elif posTag.startswith('R'):
            posTag = wn.ADV    
        elif posTag.startswith('N'):
            posTag = wn.NOUN
        else:
            continue

        wordSynst = wn.synsets(word, pos=posTag)
        if not wordSynst:
            continue  
        chosenSynst = wordSynst[0]
        sentiWordNet = swn.senti_synset(chosenSynst.name())

        # Get sentiment scores
        positiveScore = sentiWordNet.pos_score()
        negativeScore = sentiWordNet.neg_score()
        
        # Update overall sentiment scores
        sentiment_scores['pos'] += positiveScore
        sentiment_scores['neg'] += negativeScore
        
        # Calculate the composite score
        sentiment_scores['compound'] += (positiveScore - negativeScore)

    # Calculate the total sentiment for normalization
    total_sentiment = sentiment_scores['pos'] + sentiment_scores['neg']
    
    # Convert scores to percentages
    if total_sentiment > 0:
        sentiment_scores['pos'] = (sentiment_scores['pos'] / total_sentiment)
        sentiment_scores['neg'] = (sentiment_scores['neg'] / total_sentiment)
        sentiment_scores['compound'] = (sentiment_scores['compound'] / total_sentiment)
    else:
        sentiment_scores['pos'] = 0
        sentiment_scores['pos'] = 0
        sentiment_scores['composite'] = 0

    # Calculate neutral score
    sentiment_scores['neu'] = 1 - (sentiment_scores['pos'] + sentiment_scores['neg'])

    return sentiment_scores

def get_vader_scores(text):
 
    # Preprocess the text
    processed_text = cleanup_text(text)
    processed_words = preprocess_text(processed_text)
    # Join the list into a single string
    joined_text = " ".join(processed_words)
    return sia.polarity_scores(joined_text)

# Helper function to preprocess a review
def preprocess_review(review):
    sequence = [[word_index.get(word, 2) for word in review.lower().split()]]
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=padding_type)
    return padded_sequence

def get_keras_scores(text):
    sentiment_scores = {
        'pos': 0.0,
        'neg': 0.0,
        'neu': 0.0,
        'compound': 0.0
    }
    padded_sequence = preprocess_review(text)
    # Predict sentiment
    sentiment_score = keras_model.predict(padded_sequence)[0][0]
    
    #sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    
    # Convert scores to percentages
    sentiment_scores['pos'] = sentiment_score.item()
    sentiment_scores['neg'] = (1.0 - sentiment_score.item())
    sentiment_scores['compound'] = (sentiment_scores['pos'] - sentiment_scores['neg'])
    print("sentiment_scores",sentiment_scores)
    return sentiment_scores


# Route for analyzing text sentiment
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()  # Receive JSON data from the request

    # Get the text and model type from request
    text = data.get('text')
    model_type = data.get('model', 'vader').lower()  # Default to VADER if not provided

    if model_type == 'vader':
        # VADER sentiment analysis
        sentiment_scores = get_vader_scores(text)
    elif model_type == 'sentiwordnet':
        sentiment_scores = get_sentiwordnet_scores(text)
    elif model_type == 'keras':
        sentiment_scores = get_keras_scores(text)       
    else:
        # Dummy alternative model for demonstration
        # In real-world scenarios, implement other models like spaCy, TextBlob, or custom models
        sentiment_scores = {'pos': 0.0, 'neu': 0.5, 'neg': 0.5, 'composite': 0.5}  # Placeholder

    return jsonify({
        'text': text,
        'model': model_type,
        'sentiment': sentiment_scores
    })

if __name__ == '__main__':
    app.run(debug=True)
