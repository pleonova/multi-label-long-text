---
title: Multi Label Long Text
emoji: 📚
colorFrom: indigo
colorTo: gray
sdk: streamlit
app_file: app.py
pinned: false
---

Interactive version: This app is hosted on Huggingface.com/spaces

Objectvie: As the name may suggest, the goal of this app is to identify multiple relevant labels for long text.

Model: zero-shot learning - facebook/bart-large-mnli summarizer and classifier

Approach: Updating the head of the neural network, we can use the same pretrained bart model to first summarize our long text by first splitting out our long text into chunks of 1024 tokens and then generating a summary for each of the text chunks. Next, all the summaries are concanenated and the bart model is used classify the summarized text. Alternatively, one can also classify the whole text as is.



Configuration

`title`: _string_  
Display title for the Space

`emoji`: _string_  
Space emoji (emoji-only character allowed)

`colorFrom`: _string_  
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)

`colorTo`: _string_  
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)

`sdk`: _string_  
Can be either `gradio` or `streamlit`

`sdk_version` : _string_  
Only applicable for `streamlit` SDK.  
See [doc](https://hf.co/docs/hub/spaces) for more info on supported versions.

`app_file`: _string_  
Path to your main application file (which contains either `gradio` or `streamlit` Python code).  
Path is relative to the root of the repository.

`pinned`: _boolean_  
Whether the Space stays on top of your list.
