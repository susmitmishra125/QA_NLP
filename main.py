# Importing modules
from collections import defaultdict
import nltk
nltk.download('punkt') # For tokenizers
from nltk.tokenize import word_tokenize,sent_tokenize
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
import contractions
import re

# importing dataset
data = open('passage.txt','r',encoding="utf-8").read()

def preprocessing(rawReadCorpus):
		pattern = "^a-zA-Z0-9_"
		rawReadCorpus = contractions.fix(rawReadCorpus)
		# rawReadCorpus = pattern.sub('', str)
		# rawReadCorpus = rawReadCorpus.replace("’","qqq") # this is to make sure words like wouldn't are in a same token, after tokenisation @ would be replace with ’
		# rawReadCorpus = rawReadCorpus.replace("—"," ") # — is replaced with space
		# double quotes are removed as they prevented the sent_tokenize to separate sentences which ended with ” and are replaced with single " 
		# rawReadCorpus = rawReadCorpus.replace("“"," ")
		# rawReadCorpus = rawReadCorpus.replace("”"," ")

		text_sent = sent_tokenize(rawReadCorpus) # to split the sentences
		text_sent = [sent.lower() for sent in text_sent] # to convert to lowercase
		text_sent = ["".join([char for char in text if char not in string.punctuation ]) for text in text_sent] # removed punctuation
		text_sent = [word_tokenize(sent) for sent in text_sent]
		# text_sent = ["".join([char for char in text if char not in ]) for text in text_sent] # removed punctuation
		for i in range(len(text_sent)):
				sent = " ".join(text_sent[i])
				sent=re.sub(pattern,' ',sent)
				sent=sent.replace("“"," ")
				sent=sent.replace("”"," ")
				sent=sent.replace("—"," ")
				sent=sent.replace("_"," ")
				text_sent[i]=sent.split(' ')
				text_sent[i]=[i for i in text_sent[i] if i !='']
				# print(text_sent[i])
		i=len(text_sent)-1
		while(i>=0):
				if 'chapter' in text_sent[i]:
						# print(text_sent[i])
						del(text_sent[i])
				i-=1
		return text_sent
text_sent = preprocessing(data)


import torch
from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')




def answer_question(question, answer_text):
		'''
		Takes a `question` string and an `answer_text` string (which contains the
		answer), and identifies the words within the `answer_text` that are the
		answer. Prints them out.
		'''
		# ======== Tokenize ========
		# Apply the tokenizer to the input text, treating them as a text-pair.
		input_ids = tokenizer.encode(question, answer_text)

		# Report how long the input sequence is.
		# print('Query has {:,} tokens.\n'.format(len(input_ids)))

		# ======== Set Segment IDs ========
		# Search the input_ids for the first instance of the `[SEP]` token.
		sep_index = input_ids.index(tokenizer.sep_token_id)

		# The number of segment A tokens includes the [SEP] token istelf.
		num_seg_a = sep_index + 1

		# The remainder are segment B.
		num_seg_b = len(input_ids) - num_seg_a

		# Construct the list of 0s and 1s.
		segment_ids = [0]*num_seg_a + [1]*num_seg_b

		# There should be a segment_id for every input token.
		assert len(segment_ids) == len(input_ids)

		# ======== Evaluate ========
		# Run our example through the model.
		outputs = model(torch.tensor([input_ids]), # The tokens representing our input text.
										token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
										return_dict=True) 

		start_scores = outputs.start_logits
		end_scores = outputs.end_logits

		# ======== Reconstruct Answer ========
		# Find the tokens with the highest `start` and `end` scores.
		answer_start = torch.argmax(start_scores)
		answer_end = torch.argmax(end_scores)

		# Get the string versions of the input tokens.
		tokens = tokenizer.convert_ids_to_tokens(input_ids)

		# Start with the first token.
		answer = tokens[answer_start]

		# Select the remaining answer tokens and join them with whitespace.
		for i in range(answer_start + 1, answer_end + 1):
				
				# If it's a subword token, then recombine it with the previous token.
				if tokens[i][0:2] == '##':
						answer += tokens[i][2:]
				
				# Otherwise, add a space then the token.
				else:
						answer += ' ' + tokens[i]

		# print('Answer: "' + answer + '"')
		
		return (answer,start_scores[0][answer_start], end_scores[0][answer_end])


from tqdm import tqdm

def query(question,n_sent=3):
		outputs=[]
		sent_count = len(text_sent)
		ans = 0
		ans_text = 'NA'
		for i in tqdm(range(sent_count-n_sent+1)):
				segment = text_sent[i:i+n_sent]
				passage = ''
				for sent in segment:
						passage+=' '.join(sent)+'. '
				output = answer_question(question,passage)
				outputs.append((output[0],2*(output[1]*output[2])/(output[1]+output[2])))
				if 2*(output[1]*output[2])/(output[1]+output[2]) >= ans:
						ans =  2*(output[1]*output[2])/(output[1]+output[2])
						ans_text = output[0]
		outputs.sort(key=lambda x:x[1],reverse =True)
		freq = defaultdict(int)
		# print('Top 10',outputs[:10])
		for output in outputs[:10]:
			freq[output[0]]+=1
		# freq.sort(key=lambda x:x[])
		sorted(freq.items(), key=lambda item: item[1],reverse=True)
		print(freq)
		return ans_text


questions = open('question_list.txt','r').readlines()


answer = []
for i in range(len(questions)):
		# question in questions:
		questions[i]=questions[i].replace('\n','')
		print(questions[i])
		ans = query(questions[i])
		print(ans)