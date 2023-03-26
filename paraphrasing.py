import nltk
from nltk.corpus import wordnet #rule base

#naive
def paraphrase_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    new_words = []
    for word in words:
        syns = wordnet.synsets(word)
        if syns:
            new_word = syns[0].lemmas()[0].name()
            if new_word != word:
                new_words.append(new_word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

#rule based
import re
templates = [
    (re.compile(r'\b(\w+) is (?:very )?(\w+)\b', re.IGNORECASE), '\\1 is extremely \\2'),
    (re.compile(r'\b(\w+) (\w+) (\w+)\b', re.IGNORECASE), '\\3 \\2 \\1'),
    (re.compile(r'\b(the|a) (\w+) of (\w+)\b', re.IGNORECASE), '\\1 \\2 from \\3')
]

def paraphrase_with_templates(sentence):
    for pattern, replacement in templates:
        sentence = pattern.sub(replacement, sentence)
    return sentence

#data driven

#pip install git+https://github.com/PrithivirajDamodaran/Parrot.git

from parrot import Parrot
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)   
phrases = ["The dog was so excited that his tail coulsn't stop wagging, and it irritated the cat as it was being hit in the face with the said tail."]
for phrase in phrases:
  print("-"*100)
  print("Input_phrase: ", phrase)
  print("-"*100)
  para_phrases = parrot.augment(input_phrase=phrase)
  for para_phrase in para_phrases:
   print(para_phrase)
    
