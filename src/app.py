import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax


# Setup
model_path = f"pakornor/roberta-base"

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# Functions

# Preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Input preprocessing
def sentiment_analysis(text):
    text = preprocess(text)


    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)

    # Format output dictionary of scores
    labels = ['Negative', 'Neutral', 'Positive']
    scores = {l:float(s) for (l,s) in zip(labels, scores_)}

    return scores


# Gradio App
app = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox("Input tweet here:"),
    outputs="label",
    title="Sentiment Analysis of Tweets on Covid-19 Vaccines",
    description="With this App, you can type Tweets related to the Covid Vaccine and the app will rate the sentiment of the tweet..!",
    examples=[["Be careful of covid vaccination"],
              ["The vaccine can reduce your immunity to diseases"],
              ["I cant wait for the Covid Vaccine!"]]
    )

if __name__ == "__main__":
    app.launch(server_name='0.0.0.0', server_port=7860)