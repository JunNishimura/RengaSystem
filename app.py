from flask import Flask, render_template, request
import pathlib
import MeCab
import time
import random
import pandas as pd

from AI.generator import generate

class History:
    li = []
    
    def add_history(self, new_ku):
        self.li.append(new_ku)

    def reset_history(self):
        self.li = []

SEQUENCE_LENGTH = 10
yomite_history = ["AI", "PLAYER", "AI"]
season_history = History()
kigo_history = History()
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

assoc_df = pd.read_csv('./data/associative_words.csv', index_col=0)
first_keywords = assoc_df['first_keyword'].unique().tolist()
second_keywords = assoc_df['second_keyword'].unique().tolist()

ku_df = pd.read_csv('./data/ku_list.csv')
ku_list = ku_df['ku'].tolist()

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
    
    # second_keywordの方がfirst_keywordよりweightが大きい
    for s_keyword in second_keywords:
        if s_keyword in ku:
            tmp_assoc_list = assoc_df[assoc_df['second_keyword'] == s_keyword]['associative_word'].unique().tolist()
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
            tmp_assoc_list = assoc_df[assoc_df['first_keyword'] == morpheme]['associative_word'].unique().tolist()
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
    # start_time = time.time()
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
        # 句リストを全走査してベストなものを取得
        for ku in ku_list:
            score = get_ku_score(ku, assoc_words['first_keyword'], assoc_words['second_keyword'])
            if score > best_score:
                best_score = score
                next_ku = ku
        # 指定した時間が切れるまで生成
        # idx = 0
        # while int(time.time() - start_time) < time_limit:
        #     ku_generated = generate(initials[idx])
        #     score = get_ku_score(ku_generated, assoc_words['first_keyword'], assoc_words['second_keyword'])
        #     if score > best_score:
        #         best_score = score
        #         next_ku = ku_generated
        #     idx = (idx + 1) % len(initials)
    else: # 関連語・連想語がない場合は、ランダムで句を選択
        print('no associative words')
        next_ku = random.choice(ku_list)
    return next_ku

def get_season_kigo(ku: str):
    candidates = []
    kigo_df = pd.read_csv('./data/kigo.csv', index_col=0)
    
    morphemes = get_morphemes(tagger, ku)
    for morpheme in morphemes:
        match_row = kigo_df[kigo_df['読み仮名'] == morpheme]
        if len(match_row) == 1:
            # (季節, 季語)のtupleを追加
            candidates.append((match_row['季節'].tolist()[0], morpheme))
    
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        seasons = set(map(lambda x: x[0], candidates))
        kigo = ', '.join(list(map(lambda x: x[1], candidates)))
        if len(seasons) == 1:
            return (seasons.pop(), kigo)
    
    return ('', '')

@app.route("/")
def index():
    season_history.reset_history()
    kigo_history.reset_history()
    ku_history.reset_history()
    return render_template("index.html")

@app.route("/creation")
def creation():
    # 最初はAIが句を作る
    rand_val = random.randint(0, len(initials)-1)
    initial = initials[rand_val]
    AI_ku = generate(initial)
    season, kigo = get_season_kigo(AI_ku)
    season_history.add_history(season)
    kigo_history.add_history(kigo)
    ku_history.add_history(AI_ku)
    
    return render_template(
        'creation.html',
        ku_history = ku_history.li,
        yomite_history = yomite_history,
        season_history = season_history.li,
        kigo_history = kigo_history.li
    )

@app.route('/result', methods=['POST'])
def result():
    ku_former = request.form.get('input-ku__former')
    ku_latter = request.form.get('input-ku__latter')
    user_ku = ku_former + " " + ku_latter
    
    # ユーザーからの入力句をヒストリーに追加
    ku_history.add_history(user_ku)
    season, kigo = get_season_kigo(user_ku)
    season_history.add_history(season)
    kigo_history.add_history(kigo)
    # 次の句を取得
    AI_ku = generate_next_ku(ku_former, ku_latter, 10)
    season, kigo = get_season_kigo(AI_ku)
    season_history.add_history(season)
    kigo_history.add_history(kigo)
    ku_history.add_history(AI_ku)
    
    return render_template(
        'result.html', 
        ku_history = ku_history.li,
        yomite_history = yomite_history,
        season_history = season_history.li,
        kigo_history = kigo_history.li
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)