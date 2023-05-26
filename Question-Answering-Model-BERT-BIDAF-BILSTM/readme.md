# Question Answering on Squad Dataset
This project explores different models for question answering on the Squad dataset, including BiLSTM, BIDAF, BERT Uncased, and DistilBERT. We evaluate these models based on their EM and F1 scores on the train and validation datasets. We also create an ensembling model that selects the answer with the highest F1 score among the four models for each question answer pair.

## Dataset
The Squad dataset is a collection of questions and answers that require a deep understanding of a given context paragraph. The dataset consists of over 100,000 question-answer pairs that are categorized into different topics.

## Models
We used four different models for question answering:
    1. BiLSTM
    2. BIDAF
    3. BERT Cased
    4. DistilBERT

We trained each model on the Squad dataset using the training split and evaluated their performance on the validation split. We also created an ensembling model that combines the outputs of these models to improve the overall performance.

## Results
The BIDAF model achieved the highest F1 score of 0.668 on the validation dataset, while the BiLSTM model performed the worst with an F1 score of 0.285. The ensembling model achieved an F1 score of 0.786 on the validation dataset, outperforming all individual models.

## Usage
To use this project, follow these steps:

## Download the Squad dataset.
Preprocess the data to extract questions, contexts, and answers for each example.
Train each model on the training split of the dataset.
Evaluate the performance of each model on the validation split of the dataset.
Create an ensembling model that combines the outputs of these models.
Evaluate the performance of the ensembling model on the validation split of the dataset.

## Conclusion
In conclusion, this project shows that BIDAF, DistilBERT, and ensembling models are effective for question answering on the Squad dataset, while BiLSTM did not perform well. Ensembling can further improve the performance of individual models. However, more advanced models and techniques could be explored to enhance the performance of question answering systems.


Note : The code for the 4 Models are present in their respective model name folders.
Note : For all the Saved model are in savedModel inside each model folder.
Note : EMBEDDING Contains the GLOVE embedding used.
Note:  ensemble contains the ensembling model
Note: The same submission with all the saved Model are present at : 