import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def extract_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    title_tag = soup.find('h1', class_='entry-title') 
    title = title_tag.get_text().strip() if title_tag else 'No Title Found'
    
    article_tag = soup.find('div', class_='td-post-content')  
    article_text = article_tag.get_text().strip() if article_tag else 'No Content Found'

    return title, article_text

# Load the input file
input_df = pd.read_excel("C:\\Users\\Priyal\\Downloads\\20211030 Test Assignment-20240630T144446Z-001\\20211030 Test Assignment\\Input.xlsx")
url_data = input_df[['URL_ID', 'URL']]

#Directory to save text files
os.makedirs('articles', exist_ok=True)

# Extract info
for _, row in url_data.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    try:
        title, text = extract_article_text(url)
        with open(f'articles/{url_id}.txt', 'w', encoding='utf-8') as file:
            file.write(f"{title}\n{text}")
        print(f"Successfully extracted {url_id}")
    except Exception as e:
        print(f"Failed to extract {url_id}: {e}")

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import os
import re

nltk.download('punkt')

# Function to load words
def load_words(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        words = set(file.read().splitlines())
    return words

stopwords_files = [
    "C:\\Users\\Priyal\\Downloads\\20211030 Test Assignment-20240630T144446Z-001\\20211030 Test Assignment\\StopWords\\StopWords_Auditor.txt", 
    "C:\\Users\\Priyal\\Downloads\\20211030 Test Assignment-20240630T144446Z-001\\20211030 Test Assignment\\StopWords\\StopWords_Currencies.txt", 
    "C:\\Users\\Priyal\\Downloads\\20211030 Test Assignment-20240630T144446Z-001\\20211030 Test Assignment\\StopWords\\StopWords_DatesandNumbers.txt", 
    "C:\\Users\\Priyal\\Downloads\\20211030 Test Assignment-20240630T144446Z-001\\20211030 Test Assignment\\StopWords\\StopWords_Generic.txt", 
    "C:\\Users\\Priyal\\Downloads\\20211030 Test Assignment-20240630T144446Z-001\\20211030 Test Assignment\\StopWords\\StopWords_GenericLong.txt", 
    "C:\\Users\\Priyal\\Downloads\\20211030 Test Assignment-20240630T144446Z-001\\20211030 Test Assignment\\StopWords\\StopWords_Geographic.txt", 
    "C:\\Users\\Priyal\\Downloads\\20211030 Test Assignment-20240630T144446Z-001\\20211030 Test Assignment\\StopWords\\StopWords_Names.txt"
]
stopwords = set()
for file in stopwords_files:
    stopwords.update(load_words(file))

positive_words = load_words("C:\\Users\\Priyal\\Downloads\\20211030 Test Assignment-20240630T144446Z-001\\20211030 Test Assignment\\MasterDictionary\\positive-words.txt")
negative_words = load_words("C:\\Users\\Priyal\\Downloads\\20211030 Test Assignment-20240630T144446Z-001\\20211030 Test Assignment\\MasterDictionary\\negative-words.txt")

def remove_stopwords(words):
    return [word for word in words if word.lower() not in stopwords]

def compute_positive_score(words):
    return sum(1 for word in words if word in positive_words)

def compute_negative_score(words):
    return sum(1 for word in words if word in negative_words)

def compute_polarity_score(positive_score, negative_score):
    return (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

def compute_subjectivity_score(positive_score, negative_score, total_words):
    return (positive_score + negative_score) / (total_words + 0.000001)

def compute_avg_sentence_length(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return len(words) / len(sentences)

def compute_percentage_complex_words(words):
    complex_words = [word for word in words if len(word) > 2]  # Define your criteria for complex words
    return len(complex_words) / len(words)

def compute_fog_index(avg_sentence_length, percentage_complex_words):
    return 0.4 * (avg_sentence_length + percentage_complex_words)

def compute_avg_number_of_words_per_sentence(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return len(words) / len(sentences)

def compute_complex_word_count(words):
    complex_words = [word for word in words if len(word) > 2]  # Define your criteria for complex words
    return len(complex_words)

def compute_word_count(words):
    return len(words)

def compute_syllable_per_word(words):
    syllables = sum([len(re.findall('[aeiou]', word.lower())) for word in words])
    return syllables / len(words)

def compute_personal_pronouns(words):
    personal_pronouns = ['I', 'we', 'my', 'ours', 'us']
    return sum(1 for word in words if word in personal_pronouns)

def compute_avg_word_length(words):
    return sum(len(word) for word in words) / len(words)


# Directory where articles are saved
article_dir = 'articles'
results = []

# Read each article and compute metrics
for _, row in url_data.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    file_path = os.path.join(article_dir, f'{url_id}.txt')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Tokenization
        words = word_tokenize(text)
        words_filtered = remove_stopwords(words)
        
        # Compute the metrics
        positive_score = compute_positive_score(words_filtered)
        negative_score = compute_negative_score(words_filtered)
        polarity_score = compute_polarity_score(positive_score, negative_score)
        subjectivity_score = compute_subjectivity_score(positive_score, negative_score, compute_word_count(words_filtered))
        avg_sentence_length = compute_avg_sentence_length(text)
        percentage_complex_words = compute_percentage_complex_words(words_filtered)
        fog_index = compute_fog_index(avg_sentence_length, percentage_complex_words)
        avg_words_per_sentence = compute_avg_number_of_words_per_sentence(text)
        complex_word_count = compute_complex_word_count(words_filtered)
        word_count = compute_word_count(words_filtered)
        syllable_per_word = compute_syllable_per_word(words_filtered)
        personal_pronouns = compute_personal_pronouns(words_filtered)
        avg_word_length = compute_avg_word_length(words_filtered)
        
        result = [
            url_id, url, positive_score, negative_score, polarity_score, 
            subjectivity_score, avg_sentence_length, percentage_complex_words, 
            fog_index, avg_words_per_sentence, complex_word_count, word_count, 
            syllable_per_word, personal_pronouns, avg_word_length
        ]
        results.append(result)
    else:
        print(f"File {file_path} not found")


# Save results 
columns = [
    'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE', 
    'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 
    'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
]
output_df = pd.DataFrame(results, columns=columns)

# Save the results to 'Output Data Structure.xlsx'
output_df.to_excel('Output Data Structure.xlsx', index=False)

