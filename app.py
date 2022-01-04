#!/home/ubuntu/.pyenv/versions/renga/bin/python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
import pathlib
import MeCab
import time
import random
import pandas as pd

from AI.generator import generate

class History:
    ku_list = []
    
    def add_history(self, new_ku):
        self.ku_list.append(new_ku)

    def reset_history(self):
        self.ku_list = []

SEQUENCE_LENGTH = 10
yomite_history = ["AI", "PLAYER", "AI"]
ku_history = History()
app = Flask(__name__)
tagger = MeCab.Tagger("-d {}".format(pathlib.Path("./dict/")))

dakuten_dict = {
    'が': 'か', 'ぎ': 'き', 'ぐ': 'く', 'げ': 'け', 'ご': 'こ',
    'ざ': 'さ', 'じ': 'し', 'ず': 'す', 'ぜ': 'せ', 'ぞ': 'そ',
    'だ': 'た', 'ぢ': 'ち', 'づ': 'つ', 'で': 'て', 'ど': 'と',
    'ば': 'は', 'び': 'ひ', 'ぶ': 'ふ', 'べ': 'へ', 'ぼ': 'ほ',
}
initials = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふめほまみむめもらりるれろわ'

def remove_dakuten(word: str) -> str:
    new_word = ""
    for i in range(len(word)):
        new_word += dakuten_dict[word[i]] if word[i] in dakuten_dict.keys() else word[i]
    return new_word

def get_morphemes(tagger, sentence: str):
    morphemes = []
    
    node = tagger.parseToNode(sentence)
    while node:
        features = node.feature.split(",")
        if features[0] != u"BOS/EOS":
            morphemes.append(node.surface)
        node = node.next

    return morphemes

def get_ku_score(ku_candidate: str, assoc_words_first: list, assoc_words_second):
    score = 0
    
    # assoc_word_firstには3点
    for assoc_word in assoc_words_first:
        if assoc_word in ku_candidate:
            score += 3
    
    # assoc_word_secondには5点
    for assoc_word in assoc_words_second:
        if assoc_word in ku_candidate:
            score += 5
            
    if score > 0:
        print('-'*40)
        print('candidate: ', ku_candidate)
        print('first: ', assoc_words_first)
        print('second: ', assoc_words_second)
    return score
    
# 考慮する必要のある点
def get_assoc_words(ku: str, morphemes: list):
    assoc_words = {
        'first_keyword': [],
        'second_keyword': []
    }
    first_assoc_set = set()
    second_assoc_set = set()
    df = pd.read_csv('./data/associative_words.csv', index_col=0)
    first_keywords = df['first_keyword'].unique().tolist()
    second_keywords = df['second_keyword'].unique().tolist()
    
    # second_keywordの方がfirst_keywordよりweightが大きい
    for s_keyword in second_keywords:
        if s_keyword in ku:
            tmp_assoc_list = df[df['second_keyword'] == s_keyword]['associative_word'].unique().tolist()
            tmp_assoc_list2 = []
            for word in tmp_assoc_list:
                split_word = word.split('-')
                if len(split_word) == 1:
                    tmp_assoc_list2.append(remove_dakuten(word))
                elif len(split_word) > 1:
                    split_word2 = [remove_dakuten(word) for word in split_word]
                    tmp_assoc_list2 += split_word2
                    
            second_assoc_set |= set(tmp_assoc_list2)
    
    for morpheme in morphemes:
        if morpheme in first_keywords:
            tmp_assoc_list = df[df['first_keyword'] == morpheme]['associative_word'].unique().tolist()
            tmp_assoc_list2 = []
            for word in tmp_assoc_list:
                split_word = word.split('-')
                if len(split_word) == 1:
                    tmp_assoc_list2.append(remove_dakuten(word))
                elif len(split_word) > 1:
                    split_word2 = [remove_dakuten(word) for word in split_word]
                    tmp_assoc_list2 += split_word2
                    
            first_assoc_set |= set(tmp_assoc_list2)
    
    second_assoc_set -= first_assoc_set      
    assoc_words['first_keyword'] = list(first_assoc_set)
    assoc_words['second_keyword'] = list(second_assoc_set)

    return assoc_words

def generate_next_ku(cur_ku_former: str, cur_ku_latter: str, time_limit: int):
    start_time = time.time()
    next_ku = ""
    best_score = -1
    
    # 入力句に形態素解析を施す
    former_morphemes = get_morphemes(tagger, cur_ku_former)
    latter_morphemes = get_morphemes(tagger, cur_ku_latter)
    morphemes = former_morphemes + latter_morphemes
    morphemes = list(set(morphemes))
    
    # 句
    input_ku = cur_ku_former + cur_ku_latter
    
    # 関連語・連想語の取得
    assoc_words = get_assoc_words(input_ku, morphemes)
    
    # 関連語・連想語がある場合は、それらをもとに次の句として相応しい句を探索
    if len(assoc_words['first_keyword']) > 0 or len(assoc_words['second_keyword']) > 0:
        idx = 0
        # 指定した時間が切れるまで生成
        while int(time.time() - start_time) < time_limit:
            ku_generated = generate(initials[idx])
            score = get_ku_score(ku_generated, assoc_words['first_keyword'], assoc_words['second_keyword'])
            if score > best_score:
                best_score = score
                next_ku = ku_generated
            idx = (idx + 1) % len(initials)
    else: # 関連語・連想語がない場合は、ランダムで句を選択
        print('no associative words')
        rand_val = random.randint(0, len(initials)-1)
        initial = initials[rand_val]
        next_ku = generate(initial)
    return next_ku

@app.route("/")
def index():
    ku_history.reset_history()
    return render_template("index.html")

@app.route("/creation")
def creation():
    # 最初はAIが句を作る
    rand_val = random.randint(0, len(initials)-1)
    initial = initials[rand_val]
    AI_ku = generate(initial)
    # 新しい句をヒストリーに追加
    ku_history.add_history(AI_ku)
    
    return render_template(
        'creation.html',
        ku_history = ku_history.ku_list,
        yomite_history = yomite_history
    )

@app.route('/result', methods=['POST'])
def result():
    ku_former = request.form.get('input-ku__former')
    ku_latter = request.form.get('input-ku__latter')
    user_ku = ku_former + " " + ku_latter
    
    # ユーザーからの入力句をヒストリーに追加
    ku_history.add_history(user_ku)
    # 次の句を取得
    AI_ku = generate_next_ku(ku_former, ku_latter, 10)
    # 新しい句をヒストリーに追加
    ku_history.add_history(AI_ku)
    
    return render_template(
        'result.html', 
        ku_history = ku_history.ku_list,
        yomite_history = yomite_history
    )

if __name__ == '__main__':
    app.run()
