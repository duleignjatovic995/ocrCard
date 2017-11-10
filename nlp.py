"""
The purpose of nlp.py is to extract desired values from raw text we get from ocr.
"""
import re
from nltk.tag.stanford import StanfordNERTagger
import nltk


def get_person(raw_text):
    st = StanfordNERTagger('stanford-ner/english.all.3class.distsim.crf.ser.gz',
                           'stanford-ner/stanford-ner.jar')

    string = re.sub('[\.,:;\(\)\'“`‘!\?\-]', ' ', raw_text)
    tokens = nltk.word_tokenize(string)
    tags = st.tag(tokens)
    person_name_list = []
    for tag in tags:
        if tag[1] == 'PERSON':
            person_name_list.append(tag[0])

    person_name_list = list(set(person_name_list))  # Create list with unique members
    name = ' '.join(person_name_list)
    return name.strip()


def get_id(raw_text):
    ids = re.findall('[A-Za-z]{0,4}? ?[0-9]{8,}', raw_text)
    return ids
