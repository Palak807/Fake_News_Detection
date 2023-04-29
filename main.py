# Importing the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as st


def load_data():
    news = pd.read_csv('data.csv')
    X = news['text']
    y = news['label']
    return X, y


def create_pipeline():
    return Pipeline([('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
                     ('bernoulli', BernoulliNB())])


def train_model(X_train, y_train):
    model = create_pipeline()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)
    return accuracy, report, confusion


def predict_news(model, input_text):
    prediction = model.predict([input_text])
    return "Real" if prediction[0] == 0 else "Fake"


def submit():
    user_input = text_input.get("1.0", "end-1c")
    result = predict_news(model, user_input)
    result_label.config(text="The news article is: " + result)


def show_results():
    accuracy, report, confusion = evaluate_model(model, X_test, y_test)
    results_text.delete(1.0, "end")
    results_text.insert("insert",
                        f"Classification Report:\n\n{report}\nConfusion Matrix:\n\n{confusion}\n\nAccuracy Score: {accuracy:.4f}\n")


# Load data and train the model
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22)
model = train_model(X_train, y_train)

# Creating the Tkinter window
window = tk.Tk()
window.title("News Classifier")
window.geometry("800x600")
window.configure(bg='#2b2b2b')

# Adding a label and text box for user input
input_label = ttk.Label(window, text="Enter the text of the news article:", font=("Times New Roman", 14),
                        background='#2b2b2b', foreground='#ffffff')
input_label.pack(pady=20)

text_input = tk.Text(window, wrap="word", width=60, height=15, font=("Times New Roman", 12), bg='#444444', fg='#ffffff',
                     insertbackground='white', highlightbackground='#777777')
text_input.pack(padx=20, pady=20)

# Adding a button to submit the input
style = ttk.Style()
style.configure('TButton', font=('Times New Roman', 12), background='#3c8dbc', foreground='black',
                bordercolor='#3c8dbc')
style.map('TButton', background=[('active', '#3c8dbc'), ('pressed', '#2c6d9a')],
          foreground=[('active', 'black'), ('pressed', '#black')])
submit_button = ttk.Button(window, text="Classify", command=submit)
submit_button.pack(pady=10)

# Adding a label to display the result
result_label = ttk.Label(window, text="", font=("Times New Roman", 16), background='#2b2b2b', foreground='#3c8dbc')
result_label.pack(pady=20)

results_button = ttk.Button(window, text="Show Evaluation Results", command=show_results)
results_button.pack(pady=10)

# Add a text widget to display the results
results_text = st.ScrolledText(window, wrap="word", width=70, height=10, font=("Times New Roman", 12), bg='#444444',
                               fg='#ffffff', highlightbackground='#777777')
results_text.pack(padx=20, pady=20)

window.mainloop()
