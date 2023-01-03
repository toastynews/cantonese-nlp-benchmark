# cantonese-nlp-benchmark

This repo contains Cantonese word segmentation and part-of-speech tagging benchmark code and scores. It uses the metrics returned by the spaCy [Scorer](https://spacy.io/api/scorer).

## Benchmark Datasets

* [UD Cantonese HK](https://universaldependencies.org/treebanks/yue_hk/index.html)
* [UD Chinese HK](https://universaldependencies.org/treebanks/zh_hk/index.html)
* [Hong Kong Cantonese Corpus](https://pycantonese.org/data.html#built-in-data) as provided by PyCantonese

## NLP Tools Compared

* [spaCy (zh_core_web_sm)](https://spacy.io/models/zh#zh_core_web_sm) - pkuseg trained on OntoNotes
* [spaCy (zh_core_web_trf)](https://spacy.io/models/zh#zh_core_web_trf) - pkuseg trained on OntoNotes
* [pkuseg](https://github.com/lancopku/pkuseg-python)
* [cantoseg](https://github.com/ayaka14732/cantoseg) - jieba with normalized Cantonese dictionary
* [jieba (big)](https://github.com/ayaka14732/cantoseg)
* [PyCantonese](https://pycantonese.org)
* [CKIP Transformers](https://github.com/ckiplab/ckip-transformers)

## Scores
### UD Cantonese HK
|ud_hk        |pos_acc |token_f |token_p |token_r |
|:------------|--------|--------|--------|--------|
|spaCy sm     |    0.60|    0.72|    0.74|    0.69|
|spaCy trf    |    0.71|    0.72|    0.74|    0.69|
|pkuseg       |    0.61|    0.83|    0.84|    0.82|
|cantoseg     |    0.37|    0.86|    0.87|    0.85|
|jieba        |    0.40|    0.82|    0.81|    0.84|
|PyCantonese  |    0.74|    0.86|    0.87|    0.85|
|CKIP         |**0.77**|**0.89**|**0.89**|**0.90**|

### UD Chinese HK
|ud_hk        |pos_acc |token_f |token_p |token_r |
|:------------|--------|--------|--------|--------|
|spaCy sm     |    0.69|    0.82|    0.83|    0.81|
|spaCy trf    |    0.80|    0.82|    0.83|    0.81|
|pkuseg       |    0.71|    0.92|    0.93|    0.90|
|cantoseg     |    0.49|    0.84|    0.86|    0.81|
|jieba        |    0.47|    0.84|    0.87|    0.82|
|PyCantonese  |    0.65|    0.84|    0.85|    0.83|
|CKIP         |**0.81**|**0.93**|**0.93**|**0.92**|

### Hong Kong Cantonese Corpus
*PyCantonese was trained on this corpus and so this is not a fair test for it.*
|ud_hk        |pos_acc |token_f |token_p |token_r |
|:------------|--------|--------|--------|--------|
|spaCy sm     |    0.51|    0.64|    0.68|    0.60|
|spaCy trf    |    0.61|    0.64|    0.68|    0.60|
|pkuseg       |    0.39|    0.76|    0.78|    0.74|
|cantoseg     |    0.38|**0.90**|**0.93**|**0.87**|
|jieba        |    0.36|    0.80|    0.79|    0.81|
|*PyCantonese*|  *0.91*|  *0.90*|  *0.93*|  *0.87*|
|CKIP         |**0.64**|    0.84|    0.83|    0.85|

## Reproduce
Download the UD datasets, run spaCy [convert](https://spacy.io/api/cli#convert) with default options and place the files inside ./data.

To run jieba tests, the dictionaries are assumed to be inside ./jieba.

## Versions
The following software versions were used to produce the numbers above.
* spacy                     3.3.1
* pkuseg library built from [Jun 7, 2022 commit](https://github.com/lancopku/pkuseg-python/commit/071d57c7df9ac0680edda7034b47787d7c6f9184) with the [default_v2](https://github.com/lancopku/pkuseg-python/releases/download/v0.0.25/default_v2.zip) dictionary
* cantoseg dictionary built from [Aug 22, 2020 commit](https://github.com/ayaka14732/cantoseg/commit/11a4422306d57881ddaffddf0fea89554f320f32)
* jieba                    0.42.1
* pycantonese              3.4.0
* ckip-transformers        0.3.2
