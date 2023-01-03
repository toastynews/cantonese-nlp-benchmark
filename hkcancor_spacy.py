import spacy
from spacy.tokens import Doc
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.tokens import DocBin
import pprint
import pycantonese

# set up
nlp = spacy.load("zh_core_web_sm")
# nlp = spacy.load("zh_core_web_trf")
pp = pprint.PrettyPrinter(indent=4)
scorer = Scorer()

# read corpus
hkcancor = pycantonese.hkcancor()
tagged_sents = hkcancor.tokens(by_utterances=True)
examples = []
for tagged_sent in tagged_sents:
    # convert to spaCy data
    reference_words = []
    reference_spaces = []
    reference_pos = []
    for token in tagged_sent:
        reference_words.append(token.word)
        reference_spaces.append(False)
        pos_ud = pycantonese.pos_tagging.hkcancor_to_ud(token.pos)
        if pos_ud == 'V':    # pycantonese sometimes outputs this invalid tag
            pos_ud = "VERB"
        reference_pos.append(pos_ud)
    reference = Doc(nlp.vocab, words=reference_words, spaces=reference_spaces, pos=reference_pos)

    # segment with comparison system
    predicted = nlp(''.join(reference_words))

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
