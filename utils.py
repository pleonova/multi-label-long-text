import streamlit as st
import numpy as np
import plotly.express as px
import json
def plot_result(top_topics, scores):
    top_topics = np.array(top_topics)
    scores = np.array(scores)
    scores *= 100
    fig = px.bar(x=scores, y=top_topics, orientation='h', 
                 labels={'x': 'Confidence', 'y': 'Label'},
                 text=scores,
                 range_x=(0,115),
                 title='Top Predictions',
                 color=np.linspace(0,1,len(scores)),
                 color_continuous_scale='GnBu')
    fig.update(layout_coloraxis_showscale=False)
    fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')
    st.plotly_chart(fig)

def examples_load():
    with open("examples.json") as f:
        data=json.load(f)
    return data['text'],data['labels']