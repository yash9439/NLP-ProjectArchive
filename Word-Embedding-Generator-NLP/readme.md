# How to run the code:

Note: Move the code to the ../ i.e. just outside src folder

Link : https://drive.google.com/drive/folders/1iEOOdTuMFWwBAK1hXznKDvvyhUXxzcbZ?usp=sharing

    Download the files present here to get the Word_Embeddings obtained
    by the CBOW and SVD after training and get model.pk to load the model and reviews_Movies_and_TV.json 

# First Run reduceDataset.py (it picks first 50000 sentences)
    python3 reduceDataset.py

# To run SVD model and get the graph
    python3 SVD.py

# To run the train the CBOW model
    python3 CBOW_train.py

# To use the pretrained loaded CBOW model
    python3 CBOW_loaded.py

# To get the plot for CBOW model
    python3 CBOW_plot.py

# To get the similar words and plot for word "titanic"
    python3 SVD_titanic_plot.py
    python3 CBOW_titanic_plot.py