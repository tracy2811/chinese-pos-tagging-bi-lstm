# Chinese Part-of-Speech tagging with bi-LSTM attention network

Simplified Chinese Part-of-Speech tagging using bi-LSTM attention network where the word segmentation task is done:
* at the same time as the POS tagging task
* before the POS tagging task (using different libraries such as [Jieba](https://github.com/fxsjy/jieba) and [kcws](https://github.com/koth/kcws))

## Metric

F1 score is used.

## Dataset

[UD_Chinese-GSDSimp](https://github.com/UniversalDependencies/UD_Chinese-GSDSimp/tree/master) dataset from [Universal Dependencies](https://universaldependencies.org) which is free, consistent. It contains 4997 sentences and uses 15 UPOS tags out of 17 possible: ADJ, ADP, ADV, AUX, CCONJ, DET, NOUN, NUM, PART, PRON, PROPN, PUNCT, SYM, VERB, X.

## Report

* [D1-1: Proposal](./report/d1-1.md)

## Useful links 

* [awesome-chinese-nlp](https://github.com/crownpku/awesome-chinese-nlp)
* [ChineseNLP](https://github.com/didi/ChineseNLP/blob/master/docs/pos_tagging.md)

