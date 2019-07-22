# Machine Learning / Conversationnal AI Enginner - Technical Tasks
![](https://www.digitalgenius.com/wp-content/themes/digitalgenius/library/img/dg_logo_standard_deepblue.svg)
## Description
In this project I am asked to build machine learnings models for classifying text and to deploy them in a REST API (using Flask).
A full description of the project is available here: https://github.com/eliaswalyba/digital-genius/blob/master/Machine%20Learning%20Engineer%20_%20Conversational%20AI%20Engineer%20-%20Technical.pdf

## What did I built ?
I actually proposed 6 models for this project and you can find all of them [here]:https://github.com/eliaswalyba/digital-genius/blob/master/models.ipynb

1. Simple Statistical Learning Models
  1.1. Multinomial Naive Bayes Classifier
  1.2. Logistic Regression
  1.3. Linear Support Vector Machines
2. Pretrained Models
  2.1. Word2Vec
  2.2. Doc2Vec
3. A simple Neural Network Using Keras

But before building the models I took some time to make exploratory data analysis with the dataset. You can find it [here]:https://github.com/eliaswalyba/digital-genius/blob/master/eda.ipynb

I also deployed the model using [Flask]:https://palletsprojects.com/p/flask/ and all the code is available in []: https://github.com/eliaswalyba/digital-genius/tree/master/webapp

## Screenshots
![](https://raw.githubusercontent.com/eliaswalyba/digital-genius/master/Screenshot_20190722_185619.png)
![](https://raw.githubusercontent.com/eliaswalyba/digital-genius/master/Screenshot_20190722_185800.png)
![](https://raw.githubusercontent.com/eliaswalyba/digital-genius/master/Screenshot_20190722_190336.png)

## Usage
$ git clone https://github.com/eliaswalyba/digital-genius
$ conda env create -f environment.yml
$ cd digital-genius/webapp
$ python app.py

For the 2 notebooks I suggest you open them using Google Colab (https://colab.research.google.com/)

## Author
Elias W. BA

Machine Learning Enthusiast

eliaswalyba@gmail.com

😍
