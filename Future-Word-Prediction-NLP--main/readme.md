# Intro to Natural Language Processing

## Directories:
- All the dataSet are present in the ./dataSet folder
- All the Perplexity result are present inside the folder ./results
- The Checkpoint of the model is saved in ./SavedModel
- All the Codes are present inside ./sourceCode
- The Splitted Datasets are present at ./sourceCode/split

## Input Format:
- language_model.py
     $ python3 language_model.py <smoothing type> <path to corpus>
Eg=> $ python3 language_model.py k ./corpus.txt

- neural_language_model.py
     $ python3 neural_language_model.py <SavedModelPath>
Eg=> $ python3 neural_language_model.py ../models/trained_nlm.pth

Then a Prompt will come whcih will ask for a sentence as input.
After entering it, it will print the preplexity of that sentence.

### NOTE: neural_language_model.py accepts sentence of length less than 34 only.
### Also the neural_language_model.py works on kaggle but i haven't tested it outside kaggle
###      as i was unable to correctly install pytorch in my MAC M1

## SavedModel
- FirstSave.pth = for the corpus Pride-and-Prejudice-Jane-Austen
- FirstSave2.pth = for the corpus Ulysses-James-Joyce