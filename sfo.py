#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: hamzafarooq@ MABA CLASS
"""

import streamlit as st
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy
import re
from spacy import displacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.title("San Francisco Hotel Finder")

@st.cache(persist=True)
def load_data():
    df = pd.read_csv("https://datahub.io/machine-learning/iris/r/iris.csv")
    return (df)


# Define Constants
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


class SFOHotelRecs:

    def __init__(self):
        # Define embedder
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def load_data(self):
        #load the data from pickle files
        corpus = pd.read_pickle('corpus.pkl')
        corpus_embeddings = pd.read_pickle('corpus_embeddings.pkl')
        df_agg_reviews = pd.read_pickle('df_agg_reviews.pkl')
        df_agg_summary = pd.read_pickle('df_agg_summary.pkl')

        return corpus, corpus_embeddings, df_agg_reviews, df_agg_summary

    def construct_sidebar(self):
        # Construct the input sidebar for user to choose the input
        st.sidebar.markdown(
            '<p class="font-style"><b>SFO Hotel Search Criteria</b></p>',
            unsafe_allow_html=True
        )
        num_recs = st.sidebar.selectbox(
            f"Please Select the Number of Top Hotels",
            sorted(range(1, 6))
        )

        query = st.sidebar.text_area("Please Enter Your Search (Required)",'Hotels in San Francisco')

        if not query:
            st.sidebar.warning("Please fill out all required fields")

        return num_recs, query

    def plot_wordCloud(self,corpus):
        # Create and generate a word cloud image:
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="black").generate(corpus)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    def display_query(self,query):
        # display the search query
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)
        html = displacy.render(doc, style="ent")
        html = html.replace("\n", " ")
        st.markdown("<b>You Searched For:</b>", unsafe_allow_html=True)
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

    def get_recs(self):
        # Get hotel recommendations
        num_recs, query = self.construct_sidebar()
        corpus, corpus_embeddings, df_agg_reviews, df_agg_summary = self.load_data()
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest scores
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=num_recs)

        self.display_query(query)
        st.subheader("Here are the Top " + str(num_recs) + " Hotel Recommendations for your search")

        # Find the closest sentences of the corpus for each query sentence based on cosine similarity
        for score, idx in zip(top_results[0], top_results[1]):
            row_dict = df_agg_reviews.loc[df_agg_reviews['review_body'] == corpus[idx]]['hotelName'].values[0]
            summary = df_agg_summary.loc[df_agg_summary['review_body'] == corpus[idx]]['summary']
            st.write(HTML_WRAPPER.format("<b>Hotel Name:  </b>"+re.sub(r'[0-9]+', '', row_dict)+"(Score: {:.4f})".format(score)+"<br/><br/><b>Hotel Summary:  </b>"+summary.values[0]),unsafe_allow_html=True)
            self.plot_wordCloud(corpus[idx])
        print("\nCompleted corpus:")

    def construct_app(self):
        st.image('sfo.jpeg')
        st.subheader("SFO Hotel Recommendations based on customer reviews")
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
        self.get_recs()

        return self


sfo=SFOHotelRecs()
sfo.construct_app()
