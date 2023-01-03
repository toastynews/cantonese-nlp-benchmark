import spacy
from spacy.tokens import Doc
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.tokens import DocBin
import pycantonese
import pprint

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
    segmented = pycantonese.segment(doc.text)
    pos_tagged = pycantonese.pos_tag(segmented)
    pred_words = []
    pred_spaces = []
    pred_pos = []
    for (token, pos_tag) in pos_tagged:
        pred_words.append(token)
        pred_spaces.append(False)
        if pos_tag == "V":    # pycantonese sometimes outputs this invalid tag
            pred_pos.append("VERB")    
        else:
            pred_pos.append(pos_tag)        
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
