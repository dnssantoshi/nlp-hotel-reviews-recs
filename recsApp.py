import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import spacy
from spacy import displacy
import torch
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from heapq import nlargest
from string import punctuation
from sentence_transformers import SentenceTransformer, util
import scipy.spatial
import pickle as pkl
from tqdm import tqdm
import re
from transformers import pipeline
from summarizer import Summarizer
import matplotlib.pyplot as plt

# Define Constants
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

# Read reviews data
sfo_reviews = pd.read_csv('sfo_reviews.csv', header=0)

# Define stop words
stopwords = list(STOP_WORDS) + ['room','hotel','rooms','hotels']

# Define functions
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

# Define Summarizer model for provide review summary
model = Summarizer()

def summarized_review(data):
    data = data.values[0]
    return model(data, num_sentences=3)

class HotelRecs:

    def __init__(self):
        # Define embedder
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def clean_data(self):
        # Aggregate all reviews for each hotel
        df_agg_reviews = sfo_reviews.sort_values(['hotelName']).groupby('hotelName', sort=False).review_body.apply(
            ''.join).reset_index(name='review_body')

        # Generate review summary
        df_agg_summary = df_agg_reviews.copy()
        df_agg_summary['summary'] = df_agg_summary[["review_body"]].apply(summarized_review, axis=1)

        # Retain only alpha numeric characters
        df_agg_reviews['review_body'] = df_agg_reviews['review_body'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))

        # Change to lowercase
        df_agg_reviews['review_body'] = df_agg_reviews['review_body'].apply(lambda x: lower_case(x))

        # Remove stop words
        df_agg_reviews['review_body'] = df_agg_reviews['review_body'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))

        # Retain the parsed review body in the summary df
        df_agg_summary['review_body'] = df_agg_reviews['review_body']

        df_sentences = df_agg_reviews.set_index("review_body")
        df_sentences = df_sentences["hotelName"].to_dict()
        df_sentences_list = list(df_sentences.keys())

        # Embeddings
        corpus = [str(d) for d in tqdm(df_sentences_list)]
        corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True)

        # Dump to pickle file to use later for prediction
        with open("corpus.pkl", "wb") as file1:
            pkl.dump(corpus, file1)

        with open("corpus_embeddings.pkl", "wb") as file2:
            pkl.dump(corpus_embeddings, file2)

        with open("df_agg_reviews.pkl", "wb") as file3:
            pkl.dump(df_agg_reviews, file3)

        with open("df_agg_summary.pkl", "wb") as file4:
            pkl.dump(df_agg_summary, file4)

        return df_agg_summary, df_agg_reviews, corpus, corpus_embeddings
        # return  df_agg_reviews, corpus, corpus_embeddings

    def construct_app(self):
        self.clean_data()
        df_agg_summary, df_agg_reviews, corpus, corpus_embeddings = self.clean_data()
        # df_agg_reviews, corpus, corpus_embeddings = self.clean_data()

        st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="header-style"> Hotel Recommender System </p>',
            unsafe_allow_html=True
        )

        # Print summarized text
        st.markdown("Aggregated reviews")
        st.dataframe(df_agg_reviews)
        st.markdown("Aggregated summary")
        st.dataframe(df_agg_summary)
        st.markdown("Corpus")
        st.write(corpus)
        st.markdown("Corpus Embeddings")
        st.write(corpus_embeddings)

        return self

hr = HotelRecs()
hr.construct_app()