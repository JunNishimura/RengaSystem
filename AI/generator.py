import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pathlib

from AI.params import *
from AI.preprocessor import RengaPreprocessor, DakutenClassificationPreprocessor
from AI.model import RengaModel, DakutenClassifier

dakuten_dict = {
    'か': 'が', 'き': 'ぎ', 'く': 'ぐ', 'け': 'げ', 'こ': 'ご',
    'さ': 'ざ', 'し': 'じ', 'す': 'ず', 'せ': 'ぜ', 'そ': 'ぞ',
    'た': 'だ', 'ち': 'ぢ', 'つ': 'づ', 'て': 'で', 'と': 'ど',
    'は': 'ば', 'ひ': 'び', 'ふ': 'ぶ', 'へ': 'べ', 'ほ': 'ぼ'
}

def get_morpheme_sentence(tagger, ku: str) -> str:
    ku_pieces = ku.split("\t")
    sentence = []

    for piece in ku_pieces:
        node = tagger.parseToNode(piece)
        while node:
            features = node.feature.split(",")
            if features[0] != u"BOS/EOS":
                sentence.append(node.surface)

            node = node.next

    return " ".join(sentence)

def isDakutenable(word: str) -> bool:
    if word in list('かきくけこさしすせそたちつてとはひふへほ'):
        return True
    return False

def dakuten_predict(ids, model):
    with torch.no_grad():
        output = model(ids, torch.tensor([0]))
        return output.argmax(1).item()

# AIによる句の生成
def generate(initial: str):
    # 連歌モデルの構築
    pickle_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.normpath(os.path.join(pickle_dir, '../pickles/renga_df_575.pkl'))
    try:
        df = pd.read_pickle(pickle_path)
    except:
        import pickle
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
    verse_list = df.stripped_verse.tolist()
    renga_prepro = RengaPreprocessor(SEQ_LENGTH)
    renga_prepro.fit(verse_list)
    RENGA_VOCAB_SIZE = len(renga_prepro.char_to_id)
    renga_model = RengaModel(RENGA_VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS)
    renga_model.load_state_dict(torch.load(pathlib.Path('./checkpoints/renga_ckpt.pt'))['model_state_dict'])
    renga_model.eval()

    # dakuten modelの構築
    dakuten_cls_prepro = DakutenClassificationPreprocessor()
    dakuten_cls_prepro.fit()
    DAKUTEN_CLS_VOCAB_SIZE = len(dakuten_cls_prepro.char_to_id)
    dakuten_cls_model = DakutenClassifier(DAKUTEN_CLS_VOCAB_SIZE, DAKUTEN_CLS_EMBEDDING_DIM)
    dakuten_cls_model.load_state_dict(torch.load(pathlib.Path('./checkpoints/dakuten_cls_ckpt.pt'))['model_state_dict'])
    dakuten_cls_model.eval()
    dakuten_cls_model.to('cpu')

    # inference
    with torch.no_grad():
        next_char = initial    
        states = renga_model.initHidden(batch_size=1) # inference時のbatch sizeは1
        ku = initial

        # 句の生成
        while True:
            input_id = [[renga_prepro.char_to_id[next_char]]]
            input_tensor = torch.tensor(input_id, device=renga_model.device)
            logits, states = renga_model(input_tensor, states)
            probs = F.softmax(torch.squeeze(logits)).cpu().detach().numpy()
            next_id = np.random.choice(RENGA_VOCAB_SIZE, p=probs)
            next_char = renga_prepro.id_to_char[next_id]

            # 改行が出たら俳句の完成合図
            if next_char == '\n':
                break
            else:
                ku += next_char

    return ku
