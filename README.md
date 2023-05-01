# Fake News Detection Using Python
![giphy](https://media.giphy.com/media/s7CiuJbVAzAiPXRWEG/giphy.gif)


The exponential growth of digital media and the ever-increasing volume of news articles published daily have given rise to the need for efficient and accurate classification of these articles. In recent years, the spread of fake news has become a pressing issue, impacting the credibility of media sources and the integrity of public discourse. Manual classification is not only time-consuming and resource-intensive but also prone to errors and inconsistencies. As a result, there is a demand for an automated approach to classify news articles into predefined categories based on their content, which can improve content organization, enhance user experience, facilitate more targeted advertising, and detect and mitigate the spread of fake news.

This project aims to develop a Naive Bayes text classification model that can accurately predict the category of a news article using only its content, with a specific focus on identifying and flagging fake news. The model must be robust and capable of handling a diverse range of topics, maintaining high performance across various categories, and adapting to potential changes in the distribution of articles over time.

## Dataset

The dataset comprises news articles from various categories, including business, entertainment, politics, sport, tech, and fake/real news.

## Methodology

The project methodology includes data exploration, data cleaning, data visualization, feature engineering, model training, hyperparameter tuning, and performance evaluation. The Python code for the Naive Bayes text classification model can be found in the `src` folder. The code includes a pipeline that integrates the TfidfVectorizer and Bernoulli Naive Bayes classifier, allowing for efficient and accurate prediction of the category of a news article based on its content.

## Evaluation Metrics

To evaluate the effectiveness of the model, we employed various metrics, including accuracy, precision, recall, and F1-score, to provide a comprehensive evaluation of its performance across all categories.

## Installation

To run the code, please ensure that you have Python 3 installed on your machine, as well as the required libraries listed in the `requirements.txt` file.

## Usage

To run the code, simply run the main.py file.

## Conclusion

This project offers valuable insights into the development and evaluation of a Naive Bayes text classification model for news article categorization and fake news detection. The code and dataset provided in this repository can be used by news organizations, researchers, and developers to improve their content management processes and mitigate the spread of fake news.
