from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import nltk

nltk.download('punkt')

# Reference: https://discuss.huggingface.co/t/summarization-on-long-documents/920/7
def create_nest_sentences(document, token_max_length = 1024):
  nested = []
  sent = []
  length = 0
  for sentence in nltk.sent_tokenize(document):
    tokens_in_sentence = tokenizer(sentence, truncation=False, padding=False)[0] # hugging face transformer tokenizer
    length += len(tokens_in_sentence)

    if length < token_max_length:
      sent.append(sentence)
    else:
      nested.append(sent)
      sent = []
      length = 0

  if sent:
    nested.append(sent)
  return nested
  

def load_model():
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline(task='zero-shot-classification', model=model, tokenizer=tokenizer, framework='pt')
    return classifier

def classifier_zero(classifier, sequence:str, labels:list, multi_class:bool):
    outputs=classifier(sequence, labels, multi_label=multi_class)
    return outputs['labels'], outputs['scores']

