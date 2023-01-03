import spacy
from spacy.tokens import Doc
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.tokens import DocBin
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
import pprint
from os import path

# mimics https://pycantonese.org/_modules/pycantonese/pos_tagging/hkcancor_to_ud.html#hkcancor_to_ud
# references https://aclanthology.org/2022.law-1.4.pdf and https://github.com/ckiplab/ckiptagger/wiki/POS-Tags
MY_MAP = {
    "QUESTIONCATEGORY": "PUNCT",
    "PERIODCATEGORY": "PUNCT",
    "EXCLAMATIONCATEGORY": "PUNCT",
    "COMMACATEGORY": "PUNCT",
    "PAUSECATEGORY": "PUNCT",
    "PARENTHESISCATEGORY": "PUNCT",
    "DASHCATEGORY": "PUNCT",
    "COLONCATEGORY": "PUNCT",
    "ETCCATEGORY": "PUNCT",
    "SEMICOLONCATEGORY": "PUNCT",
    "DOTCATEGORY": "PUNCT",
    "SPCHANGECATEGORY": "X",
    "WHITESPACE": "X",
    "A": "ADJ", 
    "D": "ADV", 
    "Da": "ADV", 
    "Dfa": "ADV",
    "Dfb": "ADV",
    "Dk": "ADV", 
    "Di": "AUX", 
    "Caa": "CCONJ",
    "Cbb": "SCONJ",
    "Nep": "DET", 
    "Neqa": "NUM", 
    "Nes": "DET", 
    "Neu": "NUM",
    "FW": "X",  
    "Nf": "NOUN",
    "Na": "NOUN",
    "Nb": "PROPN", 
    "Nc": "NOUN", 
    "Ncd": "NOUN",
    "Nd": "NOUN", 
    "Nh": "PRON", 
    "P": "ADP", 
    "Cab": "CCONJ", 
    "Cba": "SCONJ", 
    "Neqb": "NUM", 
    "Ng": "ADP", 
    "DE": "PART",
    "I": "INTJ", 
    "T": "PART", 
    "VA": "VERB",
    "VB": "VERB", 
    "VH": "VERB",
    "VI": "VERB",
    "SHI": "AUX",
    "VAC": "VERB",
    "VC": "VERB",
    "VCL": "VERB",
    "VD": "VERB",
    "VE": "VERB",
    "VF": "VERB",
    "VG": "VERB",
    "VHC": "VERB",
    "VJ": "VERB",
    "VK": "VERB",
    "VL": "VERB",
    "V_2": "VERB",
    "Nv": "NOUN", 
    "DM": "DET",
}

# set up
nlp = spacy.load("zh_core_web_sm")
pp = pprint.PrettyPrinter(indent=4)
ws_driver  = CkipWordSegmenter(model="bert-base")
pos_driver = CkipPosTagger(model="bert-base")
scorer = Scorer()

# read corpus
doc_bin = DocBin().from_disk('data/yue_hk-ud-test.spacy')
# doc_bin = DocBin().from_disk('data/zh_hk-ud-test.spacy')
examples = []
for doc in doc_bin.get_docs(nlp.vocab):    
    # convert to spaCy data
    reference = doc    

    # segment with comparison system
    ws  = ws_driver([doc.text])
    pos = pos_driver(ws)    
    pred_words = []
    pred_spaces = []
    pred_pos = []
    pred_tags = []
    for word_ws, word_pos in zip(ws[0], pos[0]):   
        pred_words.append(word_ws)
        pred_spaces.append(False)
        pos = MY_MAP.get(word_pos) or "UNKNOWN"
        if pos == "UNKNOWN":
            print((word_ws, word_pos))
        pred_pos.append(pos)
        pred_tags.append(word_pos)    
    predicted = Doc(nlp.vocab, words=pred_words, spaces=pred_spaces, pos=pred_pos)

    # debug
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
