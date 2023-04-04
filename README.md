# Naïve Bayes Classifier for Flu Detection
This is a Python script that implements a Naïve Bayes Classifier to determine if a person has the flu or not. The classifier is designed to work with two types of input data:

1. Labels: where the data is represented as numbers (1 for Yes and 0 for No, for example) and separated by commas.
2. Words surrounded by quotes: where the data is represented as strings and surrounded by quotes.

### How to use the script 
The script can be run using any Python environment or through the command line. To run it, follow these steps:

1. Ensure that you have Python installed on your machine.
2. Open the script in your preferred editor or IDE.
3. Modify the training, outcome, and new_sample variables as per your input data.
4. Run the script.
When you run the script, it will output the probability of the new sample belonging to each class (Yes or No for labels and 'Yes' or 'No' for words surrounded by quotes), rounded to 4 decimal places.

### How it works
The script works by first calculating the probability of each class in the training data. It then calculates the probability of each feature (i.e., each data column) given each class.

When a new sample is given, the script calculates the probability of the sample belonging to each class based on the probability of the class and the probability of each feature given the class. The class with the highest probability is then chosen as the output. 
