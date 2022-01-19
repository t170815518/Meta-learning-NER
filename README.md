# Meta-NER
This is an unofficial implementation of Meta-NER[1] using PyTorch. 

Special thanks to the authors for providing core source code. 
## Dataset Description 
The tagging scheme is in IOB2 (i.e. every entity starts with 'B-')

Pre-trained word vectors are from `glove.6B.300d.txt`

The links of datasets are as shown below:

|Dataset Name|Link|
|---|---|
|Conll2003|https://deepai.org/dataset/conll-2003-english| 
|OntoNotes 5.0|https://catalog.ldc.upenn.edu/LDC2013T19| 
|WikiGold|https://github.com/NilakshanKunananthaseelan/MachineLearning/tree/main/wikigold/CONLL-format/data|
|WNUT17|https://noisy-text.github.io/2017/emerging-rare-entities.html|
|BIO_NLP_13_PC|https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BioNLP13PC-IOB|
|Cross-NER||
## Dependency 
```
numpy~=1.21.5
torch~=1.10.1
scikit-learn~=1.0.2
```
## Class Diagram 
## Experiment 
## Tutorial 
## Contact
Feel free to open a pull-request or issue for any discussion. You may also contact me via email ytang021@e.ntu.edu.sg


---
Reference:

[1]Li, J., et al. (2020). MetaNER: Named Entity Recognition with Meta-Learning. International World Wide Web Conference(
WWW). Taipei, Taiwan.
