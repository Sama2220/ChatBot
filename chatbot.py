import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from colorama import Fore, Style
import re
from fpdf import FPDF

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.loads(open('D:\\Python Projects\\intents.json').read())

# Load words, classes, and the trained model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Sample shelf data
shelves = {
    "milk": 1, "yogurt": 1, "cookies": 1, "biscuits": 1,
    "bread": 2, "butter": 2, "chips": 2, "eggs": 3, "egg": 3,
    "orange juice": 3, "salsa": 3, "cheese": 4, "tomatoes": 4, 
    "tomato": 4, "avocado": 4, "apples": 5, "apple": 5, 
    "potatoes": 5, "potato": 5, "lettuce": 5, "bananas": 6,
    "banana": 7, "carrots": 6, "carrot": 6, "spinach": 6,
    "chicken": 7, "beef": 7, "soup": 7, "rice": 8, "fish": 8,
    "crackers": 8, "pasta": 9, "shrimp": 9, "pineapple": 9,
    "cereal": 10, "ice cream": 10, "watermelon": 10
}

context = {}

# Function to clean up and tokenize input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Create a bag of words for the input sentence
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Predict the class of the input sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Get the response based on the predicted class and user message
def get_response(intents_list, intents_json, user_message):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    shelf_info = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            if tag == "provide_shelves":
                items = re.split(r',\s*|\s+and\s+|\s+&\s+|\s+', user_message)
                shelf_info = "\n".join([f"{item.strip()}: Shelf {shelves.get(item.strip().lower(), 'Not found')}" for item in items])
                result = random.choice(i['responses']).strip() + "\n" + shelf_info
                create_pdf(shelf_info)
            else:
                result = random.choice(i['responses'])
            break
    return result

# Create a PDF with the shelf numbers for the items
def create_pdf(shelf_info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Shelf Numbers for Your Items", ln=True, align='C')
    pdf.ln(10)  # Add a line break
    for line in shelf_info.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True, align='L')
    pdf.output("shelf_numbers.pdf")
    print(Fore.YELLOW + "PDF generated: shelf_numbers.pdf" + Style.RESET_ALL)

print()
print(Fore.YELLOW + "GO! Bot is Running!" + Style.RESET_ALL)
print("-------------------")
print()

# Main loop to interact with the customer
while True:
    message = input(Fore.LIGHTGREEN_EX + "Customer: " + Style.RESET_ALL)  
    ints = predict_class(message)
    
    if len(ints) > 0 and ints[0]['intent'] == 'goods_list':
        context['intent'] = 'provide_shelves'
        response = get_response(ints, intents, message)
    elif 'intent' in context and context['intent'] == 'provide_shelves':
        ints = [{'intent': 'provide_shelves', 'probability': '1.0'}]
        response = get_response(ints, intents, message)
        context.clear()
    elif ints[0]['intent'] == 'goodbye':
        response = get_response(ints, intents, message)
        context.clear()
        print(Fore.BLUE + "Chatbot: " + Fore.CYAN + response + Style.RESET_ALL)
        print()
        print(Fore.YELLOW + "GO! Bot is Running!" + Style.RESET_ALL)
        print("-------------------")
        print()
        continue
    else:
        response = get_response(ints, intents, message)

    print(Fore.BLUE + "Chatbot: " + Fore.CYAN + response + Style.RESET_ALL) 
    print()
