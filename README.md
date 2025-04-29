Sarcasm Detection Using Bidirectional LSTMs:
This project aims to build a sarcasm detection system for news headlines using Bidirectional Long Short-Term Memory (Bi-LSTM) networks. The model is trained to classify headlines as either sarcastic or non-sarcastic. We use a pre-trained GloVe word embedding to map words to vector representations, followed by an LSTM network to make predictions.

Project Structure:
The project is organized into the following sections:

Data Preparation

Text Preprocessing

Model Training

Model Evaluation

Results Visualization

Dependencies:
This project requires the following Python libraries:

tensorflow (for building and training the model)

numpy (for numerical operations)

pandas (for data manipulation)

matplotlib and seaborn (for data visualization)

scikit-learn (for machine learning tools)

nltk (for text processing)

Install the dependencies using:
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn nltk

Dataset:
The dataset used in this project is the Sarcasm Headlines Dataset. It contains headlines from The Onion and HuffPost with labels indicating whether the headline is sarcastic (1) or not (0). The dataset is stored in JSON format.


Data Preprocessing:
The steps below describe how the data is preprocessed:

Loading the Dataset: The dataset is loaded into a pandas DataFrame using the pd.read_json() function.

Text Tokenization: We use Tokenizer from Keras to tokenize the text (convert words into numeric indices) and create a vocabulary.

Padding: Text sequences are padded to a uniform length using pad_sequences. This ensures that the input to the model has a consistent shape.

Train-Test Split: The dataset is split into training and testing sets using train_test_split from scikit-learn. We allocate 80% of the data for training and 20% for testing.

Model Architecture:
We use a Bi-directional LSTM model for sarcasm detection. The architecture consists of:

Embedding Layer: Uses pre-trained GloVe embeddings to convert words into dense vector representations. We load the GloVe embeddings from a file (glove.6B.50d.txt).

Bidirectional LSTM Layers: Two layers of Bidirectional LSTM are used to capture the sequential patterns in the headlines.

Dense Layer: A fully connected layer with a sigmoid activation function to predict the probability of a headline being sarcastic.

The model is compiled using the Adam optimizer and binary crossentropy loss function for binary classification.

Training the Model:
We train the model using the fit method for a number of epochs (10 in this case). The training data is divided into batches, and we use validation data to monitor the model's performance during training.

Model Evaluation:
Once trained, we evaluate the model on the test set using:

Accuracy: Measures the percentage of correct predictions.

Confusion Matrix: Displays the true positives, false positives, true negatives, and false negatives.

Classification Report: Provides metrics like precision, recall, and F1-score for both classes (sarcastic and non-sarcastic).

Results Visualization:
We visualize the following:

Training and Validation Accuracy: The accuracy over the epochs for both the training and validation sets is plotted.

Confusion Matrix: A heatmap of the confusion matrix to visualize the performance of the model.

Predicted vs. Actual: A comparison of predicted and actual labels for the test set.

Future Work:
The model can be further improved by:

Hyperparameter Tuning: Experimenting with different LSTM units, batch sizes, and epochs.

Advanced Embeddings: Using more advanced word embeddings like FastText or BERT.

Fine-tuning the model: Fine-tuning the GloVe embeddings by making them trainable during model training.
