# Data4StylizedS2S
The repository contains resources of our paper published at AAAI 2020 ``A Dataset for Low-Resource Stylized Sequence-to-Sequence Generation "

I strongly recommend to use the MT data set to evaluate your Stylized Sequence-to-Sequence Generation system. 

## Machine Translation Formality Corpus
The data is uploaded to Google drive, https://drive.google.com/file/d/1TX5-7T4qLrM0PdZVvnDb-ePqo4evJjb0/view?usp=sharing. We also uploaded our predictions to the URL, so you can use your  tool to evaluate different methods. According to our evaluation scripts, the results are as follows


|      |Formality|Fluency|Overall  (BLEU Score)       | 
| ------------- |:-------------:|:-------------:|:-------------:|
| Base  | 0.55 | 3.61| 33.47 |
| Pivot_Rule     | 0.68|3.58| 37.83   | 
| Pivot_Model | 0.679|3.64|38.75      | 
| Teacher Student | 0.768|3.60| 40.07    | 
| backtranslation | 0.707|3.59|40.68    | 

For more details, please read our paper.

## Twitter Conversation Formality Corpus

Conversation data can be found at https://drive.google.com/file/d/1KirCpHgCqtBsD_5TY1OOaKdz-6A48Nt6/view?usp=sharing. Regarding training corpus, we only provide twitterID and userID due to legal issue. validation and test files are merged into one file, where 0-1000 are validation files and 1000-2000 are test files. Please ignore those rewriting results are none.

Because conversation outputs are extremely hard to evaluate, we do not recommand to evaluate your predictions with automatic metrics. However, we still provide our evaluation results. 


|      |Formality|Fluency|Overall  (Human Evalution)       | 
| ------------- |:-------------:|:-------------:|:-------------:|
| Base  | 0.506 | 3.25| 2.21|
| Pivot_Rule     | 0.583|3.69| 2.41   | 
| Pivot_Model | 0.807|3.72|2.51     | 
| Teacher Student | 0.862|3.73| 2.87    | 
| backtranslation | 0.844|3.72|3.16   | 

