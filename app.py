# Reference: https://huggingface.co/spaces/team-zero-shot-nli/zero-shot-nli/blob/main/app.py

from os import write
# import pandas as pd
from typing import Sequence
import streamlit as st

from models import create_nest_sentences, load_summary_model, summarizer_gen, load_model, classifier_zero
from utils import plot_result, examples_load, example_long_text_load
# from utils import plot_result, examples_load, example_long_text_load, to_excel
import json


summarizer = load_summary_model()   
classifier = load_model()
ex_text, ex_license, ex_labels = examples_load()
ex_long_text = example_long_text_load()


if __name__ == '__main__':
    st.header("Summzarization & Multi-label Classification for Long Text")
    st.write("This app summarizes and then classifies your long text with multiple labels.")


    with st.form(key='my_form'):
        example_text = "[Excerpt from Project Gutenberg: Frankenstein]\n" + ex_long_text + "\n\n" + ex_license
        text_input = st.text_area("Input any text you want to classify here:", example_text)

        if text_input == example_text:
            text_input = ex_long_text

        # minimum_tokens = 30
        # maximum_tokens = 100
        labels = st.text_input('Possible label topic names (comma-separated):',ex_labels, max_chars=1000)
        labels = list(set([x.strip() for x in labels.strip().split(',') if len(x.strip()) > 0]))
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if len(labels) == 0:
            st.write('Enter some text and at least one possible topic to see predictions.')

        # For each body of text, create text chunks of a certain token size required by the transformer
        nested_sentences = create_nest_sentences(document = text_input, token_max_length = 1024)
        st.write("Number of Text Chunks to Summarize")
        st.markdown(len(nested_sentences))
        summary = []
        # For each chunk of sentences (within the token max), generate a summary
        for n in range(0, len(nested_sentences)):
            text_chunk = " ".join(map(str, nested_sentences[n]))
            st.markdown("#### Text Chunks,", n/len(nested_sentences))
            st.markdown(text_chunk)

            chunk_summary = summarizer_gen(summarizer, sequence=text_chunk, maximum_tokens = 300, minimum_tokens = 20)
            summary.append(chunk_summary) 
            st.markdown("#### Partial Summary")
            st.markdown(chunk_summary)
            # Combine all the summaries into a list and compress into one document, again
            final_summary = " ".join(list(summary))

        # final_summary = summarizer_gen(summarizer, sequence=text_input, maximum_tokens = 30, minimum_tokens = 100)
        st.markdown("### Full Text Summary")
        st.markdown(final_summary)

        st.markdown("### Top Label Predictions")
        top_topics, scores = classifier_zero(classifier, sequence=final_summary, labels=labels, multi_class=True)
        plot_result(top_topics[::-1][:], scores[::-1][:])

        # df_xlsx = to_excel(df = pd.DataFrame(scores))
        # st.download_button(label='📥 Download Current Result',
        #                         data=df_xlsx ,
        #                         file_name= 'df_test.xlsx')
