# Importing modules
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
import torch
from collections import Counter
from tqdm import tqdm
import re
import contractions
from nltk.corpus import stopwords
import string
from collections import defaultdict
import nltk
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')  # For tokenizers
nltk.download('stopwords')


# importing dataset
data = open('passage.txt', 'r', encoding="utf-8").read()
questions = open('question_list.txt', 'r').readlines()

# preprocess for modern method


def preprocessing(rawReadCorpus, complete_preprocess=False):
    pattern = "^a-zA-Z0-9_"
    rawReadCorpus = contractions.fix(rawReadCorpus)
    text_sent = sent_tokenize(rawReadCorpus)  # to split the sentences
    text_sent = [sent.lower() for sent in text_sent]  # to convert to lowercase
    text_sent = ["".join([char for char in text if char not in string.punctuation])
                 for text in text_sent]  # removed punctuation
    text_sent = [word_tokenize(sent) for sent in text_sent]
    # text_sent = ["".join([char for char in text if char not in ]) for text in text_sent] # removed punctuation
    for i in range(len(text_sent)):
        sent = " ".join(text_sent[i])
        sent = re.sub(pattern, ' ', sent)
        sent = sent.replace("“", " ")
        sent = sent.replace("”", " ")
        sent = sent.replace("—", " ")
        sent = sent.replace("_", " ")
        text_sent[i] = sent.split(' ')
        text_sent[i] = [i for i in text_sent[i] if i != '']
        # print(text_sent[i])
    if complete_preprocess:
        ps = nltk.porter.PorterStemmer()
        for i in range(len(text_sent)):
            text_sent[i] = [ps.stem(j) for j in text_sent[i]]
    i = len(text_sent)-1
    while(i >= 0):
        if 'chapter' in text_sent[i]:
            # print(text_sent[i])
            del(text_sent[i])
        i -= 1
    return text_sent


def answer_question(question, answer_text, model, tokenizer):
    '''
    Takes a question string and an answer_text string (which contains the
    answer), and identifies the words within the answer_text that are the
    answer. Prints them out.
    '''
    input_ids = tokenizer.encode(question, answer_text)
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    outputs = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                    # The segment IDs to differentiate question from answer_text
                    token_type_ids=torch.tensor([segment_ids]),
                    return_dict=True)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokens[answer_start]
    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]
    return (answer, start_scores[0][answer_start], end_scores[0][answer_end])


def query(question, text_sent, model, tokenizer, n_sent=3):
    outputs = []
    sent_count = len(text_sent)
    ans = 0
    ans_text = 'NA'
    for i in tqdm(range(sent_count-n_sent+1)):
        segment = text_sent[i:i+n_sent]
        passage = ''
        for sent in segment:
            passage += ' '.join(sent)+'. '
        output = answer_question(question, passage, model, tokenizer)

        outputs.append((output[0], 2*(output[1]*output[2])/(output[1]+output[2])))
        if(len(outputs)>10):
            outputs.sort(key=lambda x: x[1], reverse=True)
            outputs=outputs[:10]

    cnt = Counter()
    for output in outputs:
        cnt[output[0]] += 1
    return cnt.most_common()[0][0]


def modern():
    text_sent = preprocessing(data)
    model = BertForQuestionAnswering.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    for i in range(len(questions)):
        # question in questions:
        questions[i] = questions[i].replace('\n', '')
        print(questions[i])
        ans = query(questions[i], text_sent=text_sent,model = model,tokenizer = tokenizer)
        print(ans)

from numpy import dot
from numpy.linalg import norm

# cos_sim = dot(a, b)/(norm(a)*norm(b))
def traditional():
    print("starting preprocessing")
    text_sent = preprocessing(data,complete_preprocess=False)
    text_sent = [' '.join(sent) for sent in text_sent]
    print("complete preprocessing")
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    
    sentence_vectors = vectorizer.fit_transform(text_sent).toarray()

    for question in questions:
        ans = "NA"
        max_cos = 0
        q_vec = vectorizer.transform([question]).toarray()[0]
        for i in range(len(text_sent)):
            ans_vec = sentence_vectors[i]
            cos_sim = dot(ans_vec,q_vec)/(norm(ans_vec)*norm(q_vec)+1e-9)
            if cos_sim>max_cos:
                # print(sum(q_vec),sum(ans_vec))
                ans = text_sent[i]
                max_cos = cos_sim
        print(question)
        print(ans,'\n')
    # print(sentence_vectors.toarr)
print("Starting for modern method")
modern()
print("Starting for traditional method")
traditional()