import spacy
from spacy.tokens import Doc
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.tokens import DocBin
import pprint

# set up
nlp = spacy.load("zh_core_web_sm")
# nlp = spacy.load("zh_core_web_trf")
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
    predicted = nlp(doc.text)

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
