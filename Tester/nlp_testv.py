import ocr
import re
from nltk.tag.stanford import StanfordNERTagger
import nltk

# todo Find ids PROBLEM: Vraca vise id-a.
# todo Find names PROBLEM: Dependents upada kao name
# todo Find insurance / template matching!

img1 = 'cards/14 HIP Prime EPO.png'
img2 = 'cards/arch1.png'
img3 = 'cards/prim1.png'
img4 = 'cards/empire2.png'
img5 = 'cards/prim2.png'
img6 = 'cards/prim3.png'
img7 = 'cards/visa2.png'

IMAGE_PATH = img1

st = StanfordNERTagger('stanford-ner/english.all.3class.distsim.crf.ser.gz',
                           'stanford-ner/stanford-ner.jar')


def get_person1(raw_text):
    for sent in nltk.sent_tokenize(raw_text):
        # print("Sent", sent)
        tokens = nltk.tokenize.word_tokenize(sent)
        # print("Word", tokens)
        tags = st.tag(tokens)
        person_name_list = []
        for tag in tags:
            if tag[1] == 'PERSON':
                person_name_list.append(tag[0])

        person_name_list = list(set(person_name_list))  # Create list with unique members
        name = ' '.join(person_name_list)
        return name.strip()


def get_person(raw_text):
    string = re.sub('[\.,:;\(\)\'“`‘!\?\-]', ' ', raw_text)
    tokens = nltk.word_tokenize(string)
    print("Words: ", tokens)
    tags = st.tag(tokens)
    person_name_list = []
    for tag in tags:
        print(tag)
        if tag[1] == 'PERSON':
            person_name_list.append(tag[0])

    person_name_list = list(set(person_name_list))  # Create list with unique members
    name = ' '.join(person_name_list)
    return name.strip()


def get_id(raw_text):
    ids = re.findall('[A-Za-z]{0,4}? ?[0-9]{8,}', raw_text)
    return ids


# TEST PART #

def test_ner(raw_text):
    # print(raw_text)
    #
    # print('member name: ', get_person(raw_text))
    # print('member id: ', get_id(raw_text))
    # english.muc.7class.distsim.crf.ser.gz
    # 'stanford-ner/english.all.3class.distsim.crf.ser.gz'
    st = StanfordNERTagger('stanford-ner/english.muc.7class.distsim.crf.ser.gz',
                           'stanford-ner/stanford-ner.jar')
    for sent in nltk.sent_tokenize(raw_text):
        tokens = nltk.tokenize.word_tokenize(sent)
        tags = st.tag(tokens)
        person_name_list = []
        for tag in tags:
            # if tag[1] == 'LOCATION':
            #     print(tag)
            print(tag)


raw_text = ocr.get_text(IMAGE_PATH)\
    .replace('SAMPLE', 'JOHN')\
    .replace('Member Name', 'JOHN JONSON')\
    .replace('CARD', 'JONSON')

print(raw_text)
print('TEST: ')
print(get_person(raw_text))
# print(get_person1(raw_text))
print(get_id(raw_text))
