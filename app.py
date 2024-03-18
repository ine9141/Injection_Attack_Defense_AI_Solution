from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import WordPunctTokenizer
import pickle

app = Flask(__name__)

# 저장된 모델 불러오기
sql_model = load_model('sql_model.h5')
xss_model = load_model('xss_model.keras')
commend_model = load_model('commend_model.keras')

tokenizer = WordPunctTokenizer()

# sql_index 딕셔너리 불러오기
with open('sql_index.pkl', 'rb') as f:
    sql_index = pickle.load(f)

# xss_index 딕셔너리 불러오기
with open('xss_index.pkl', 'rb') as f:
    xss_index = pickle.load(f)

# comment_index 딕셔너리 불러오기
with open('commend_index.pkl', 'rb') as f:
    commend_index = pickle.load(f)


def sql_valid(input_text):
    encoded_text = [[sql_index.get(token, 0) for token in tokenizer.tokenize(input_text.lower())]]
    padded_text = pad_sequences(encoded_text, maxlen=114)
    score = float(sql_model.predict(padded_text))
    return score


def commend_valid(input_text):
    encoded_text = [[commend_index.get(token, 0) for token in tokenizer.tokenize(input_text.lower())]]
    padded_text = pad_sequences(encoded_text, maxlen=114)
    score = float(sql_model.predict(padded_text))
    return score


def xss_valid(input_text):
    encoded_text = [[xss_index.get(token, 0) for token in tokenizer.tokenize(input_text.lower())]]
    padded_text = pad_sequences(encoded_text, maxlen=114)
    score = float(sql_model.predict(padded_text))
    return score


@app.route('/')
def index():
    data = request.get_json()

    # 사용자의 입력 받기
    title = data.get('title')
    content = data.get('content')

    response = {
        'title': title,
        'content': content,
        'is_sql_injection': False
    }

    return jsonify(response), 200


@app.route('/login')
def login():
    data = request.get_json()

    # 아이디 입력
    id = data.get('id')
    pw = data.get('pw')

    # 유효성 검사
    id_score = sql_valid(id)
    pw_score = sql_valid(pw)

    response = {
        'id': id,
        'pw': pw,
        'valid': id_score > 0.5 or pw_score > 0.5
    }

    return jsonify(response), 200


@app.route('/board', methods=['POST'])
def valid():
    data = request.get_json()

    # 사용자의 입력 받기
    title = data.get('title')
    content = data.get('content')

    # 모델을 활용한 유효성 검사
    title_score = xss_valid(title)
    content_score = xss_valid(content)
    commend_score0 = commend_valid(title)
    commend_score1 = commend_valid(content)

    response = {
        'title': title,
        'content': content,
        'valid': title_score > 0.5 or content_score > 0.5 or commend_score0 > 0.5 or commend_score1 > 0.5
    }

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=False)
