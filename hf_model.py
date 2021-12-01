from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import nltk

nltk.download('punkt')



def load_model():
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline(task='zero-shot-classification', model=model, tokenizer=tokenizer, framework='pt')
    return classifier

def classifier_zero(classifier, sequence:str, labels:list, multi_class:bool):
    outputs=classifier(sequence, labels, multi_label=multi_class)
    return outputs['labels'], outputs['scores']

