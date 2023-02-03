# Fake-News-Classification-Using-LSTM

In this project, we have used various Natural language processing techniques with LSTM model to classify fake news articles using Tensorflow and Sci-kit libraries from python. 

Download dataset from kaggle https://www.kaggle.com/competitions/fake-news/data?select=train.csv

# Table of contents

- [Table of contents](#table-of-contents)
- [Implementation](#Implementation)
- [Adding new features](#adding-new-features)

Python 3.9 was used to create the application files. Before running the files, it must be ensured that Python 3.9 and the following libraries are installed.

| Library  | Task |
| ------------- | ------------- |
| [Numpy](https://numpy.org/install/)  | Mathematical Operations  |
| [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)  | Data Analysis Tools  |
| [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)  | Visualizations  |
| [Sklearn](https://scikit-learn.org/stable/install.html)  | Machine Learning Library  |
| [Tensorflow](https://www.tensorflow.org/install)  | Modelling  |
| [NLTK](https://www.nltk.org/data.html)  | NLP Library  |


# [Implementation](#table-of-contents)

**Download dataset from data folder**


1. Data Cleaning for Analysis: In this section, we will clean our dataset to do some analysis:
    - Perform null value imputation.
    - Remove stop words.
    - Remove special characters.
    - Drop unused rows and columns.
    - Apply stemming.

2. Explorative Data Analysis: In this section, we will perform:
    - Statistical Analysis of the text.
    - Word Cloud Visualizations of text analysis.

3. Building a LSTM Classifier
    - Data Preperation: In this section splitting of dataset into training and testing is done. 
    - Tokenizing the Dataset: One Hot Representation and post padding is applied to fix a sentence length to fix the input on the dataset. 
    - Training the model: Embedding layer , LSTM , Dense layers are added to Sequential model and binary cross entropy, adam optimer, accuracy as metrics are used to configure the model for training.
    - Model Evaluation : Accuracy score and confusion matrix are evaluated on the test dataset.

Each of these steps contained in fake-news-classification-using-lstm.ipynb file. The file with the ipynb extension has the advantage of saving the state of the last run of that file and the screen output.

Thus, screen output can be seen without re-running the files. Files with the ipynb extension can be run using the [jupyter notebook](https://jupyter.org/) program. When running the codes, the sequence numbers in the filenames should be followed.

Because the output of almost every step is the prerequisite for the operation of the next step. 


### [Adding new features](#table-of-contents)
The overall accuracy of our trained model when classifying articles is 0.9074. It should be noted that this number represents perfect classification. In fact, other models that focus on binary classification of Fake news label may achieve higher accuracy of perfect classification.

In furter improvements we can train our model with Different Classification algorithms like Logsitic Regression, Naive Bayes, SVM, KNN, Random Forest and AdaBoost and based on the comparision analysis on performance metrics we can decide the best algorithm for Fake-news classification.
