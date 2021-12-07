import streamlit as st
import numpy as np
import plotly.express as px
import json
# from pyxlsb import open_workbook as open_xlsb
# import pandas as pd
# from io import BytesIO

def plot_result(top_topics, scores):
    top_topics = np.array(top_topics)
    scores = np.array(scores)
    scores *= 100
    fig = px.bar(x=scores, y=top_topics, orientation='h', 
                 labels={'x': 'Confidence', 'y': 'Label'},
                 text=scores,
                 range_x=(0,115),
                 # title='Top Predictions',
                 color=np.linspace(0,1,len(scores)),
                 color_continuous_scale='GnBu')
    fig.update(layout_coloraxis_showscale=False)
    fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')
    st.plotly_chart(fig)

def examples_load():
    with open("examples.json") as f:
        data=json.load(f)
    return data['text'], data['long_text_license'], data['labels']

def example_long_text_load():
    with open("example_long_text.txt", "r") as f:
        text_data = f.read()
    return text_data

# # Reference: https://discuss.streamlit.io/t/download-button-for-csv-or-xlsx-file/17385
# def to_excel(df):
#     output = BytesIO()
#     writer = pd.ExcelWriter(output, engine='xlsxwriter')
#     df.to_excel(writer, index=False, sheet_name='Sheet1')
#     workbook = writer.book
#     worksheet = writer.sheets['Sheet1']
#     format1 = workbook.add_format({'num_format': '0.00'}) 
#     worksheet.set_column('A:A', None, format1)  
#     writer.save()
#     processed_data = output.getvalue()
#     return processed_data
