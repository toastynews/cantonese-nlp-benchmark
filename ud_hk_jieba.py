import spacy
from spacy.tokens import Doc
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.tokens import DocBin
import pycantonese
import jieba
import pprint
from os import path

jieba.set_dictionary('jieba/merged_dict.txt')    # cantoseg
# jieba.set_dictionary('jieba/dict.txt.big')    # jieba big handles tradidional characters better
jieba.initialize()

import jieba.posseg as pseg

# mimics https://pycantonese.org/_modules/pycantonese/pos_tagging/hkcancor_to_ud.html#hkcancor_to_ud
# references https://blog.pulipuli.info/2017/11/fasttag-identify-part-of-speech-in.html for the unhandled tags
MY_MAP = {
    "ZG": "ADV",
    "ENG": "X",
    "UD": "ADV",
    "UV": "ADV",
    "UJ": "ADV",
    "UL": "ADV",
    "UZ": "ADV",
    "NRT": "PROPN",
    "MQ": "PRON",
    "DF": "VERB",
    "NRFG": "NOUN",
    "VQ": "VERB",
    "RR": "PRON",
    "VX": "VERB",
}

# set up
nlp = spacy.load("zh_core_web_sm")
pp = pprint.PrettyPrinter(indent=4)
scorer = Scorer()

# read corpus
doc_bin = DocBin().from_disk('data/yue_hk-ud-test.spacy')
# doc_bin = DocBin().from_disk('data/zh_hk-ud-test.spacy')
examples = []
for doc in doc_bin.get_docs(nlp.vocab):
    # convert to spaCy data
    reference = doc    

    # segment with comparison system
    words = pseg.cut(doc.text)
    pred_words = []
    pred_spaces = []
    pred_pos = []
    pred_tags = []
    for w in words:
        pred_words.append(w.word)
        pred_spaces.append(False)
        upper_flag = w.flag.upper()
        pos = MY_MAP.get(upper_flag) or "UNKNOWN"
        if pos == "UNKNOWN":
            pos = pycantonese.pos_tagging.hkcancor_to_ud(upper_flag)
        pred_pos.append(pos)
        pred_tags.append(w.flag)       
    predicted = Doc(nlp.vocab, words=pred_words, spaces=pred_spaces, pos=pred_pos)

    # debug
    # predicted = Doc(nlp.vocab, words=pred_words, spaces=pred_spaces)
    # for token in reference:
    #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #         token.shape_, token.is_space, token.is_alpha, token.is_stop)
    # for token in predicted:
    #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #         token.shape_, token.is_space, token.is_alpha, token.is_stop)

    example = Example(predicted, reference)
    # scores = scorer.score_tokenization([example])
    # pp.pprint(scores)
    examples.append(example)

# calculate scores
scores = scorer.score(examples)
pp.pprint(scores)
