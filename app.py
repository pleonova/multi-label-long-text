# Reference: https://huggingface.co/spaces/team-zero-shot-nli/zero-shot-nli/blob/main/app.py

from os import write
from typing import Sequence
import streamlit as st
from hf_model import classifier_zero, load_model
from utils import plot_result, examples_load
import json

classifier=load_model()
ex_text,ex_labels=examples_load()


if __name__ == '__main__':
    st.header("Multi-label Classification for Long Text")
    st.write("This app identifies multiple relevant labels for your long text.")


    with st.form(key='my_form'):
        text_input = st.text_area("Input any text you want to classify here:",ex_text)
        labels = st.text_input('Write any topic keywords you are interested in here (separate different topics with a ","):',ex_labels, max_chars=1000)
        labels = list(set([x.strip() for x in labels.strip().split(',') if len(x.strip()) > 0]))
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if len(labels) == 0:
            st.write('Enter some text and at least one possible topic to see predictions.')
        top_topics, scores = classifier_zero(classifier, sequence=text_input, labels=labels, multi_class=True)
        plot_result(top_topics[::-1][-10:], scores[::-1][-10:])
