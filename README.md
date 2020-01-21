# Data4StylizedS2S
The repository contains resources of our paper published at AAAI 2020 ``A Dataset for Low-Resource Stylized Sequence-to-Sequence Generation "

I strongly recommend to use the MT data set to evaluate your Stylized Sequence-to-Sequence Generation system. 

## Machine Translation Formality Corpus
The data is uploaded to Google drive, https://drive.google.com/file/d/1TX5-7T4qLrM0PdZVvnDb-ePqo4evJjb0/view?usp=sharing. We also uploaded our predictions to the URL, so you can use your  tool to evaluate different methods. According to our evaluation scripts, the results are as follows


|      |Formality|Fluency|Overall         | 
| ------------- |:-------------:|:-------------:|:-------------:|
| Base  | 0.55 | 3.61| 33.47 |
| Pivot_Rule     | 0.68|3.58| 37.83   | 
| Pivot_Model | 0.679|3.64|38.75      | 
| Teacher Student | 0.768|3.60| 40.07    | 
| backtranslation | 0.707|3.59|40.68    | 

For more details, please read our paper.
