from telegram import Update
import os
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import random
import json
import torch
import time  # Import the time module

from model import NeuralNet
from nltk_utils import tokenize, bag_of_words, stem

# API TELEGRAM
load_dotenv()
TOKEN = os.getenv('BOT_TOKEN_API')
BOT_USERNAME = os.getenv('BOT_USERNAME')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# FUNCTION COMMANDS
async def start_command(update: Update, context):
    await update.message.reply_text('Selamat Datang Di Bot Catering Anggur, Bot ini hanya menyediakan informasi Terkait Catering Anggur, Segala bentuk pemesanan dan pembayaran melaui Whatsapp atau ke Catering Anggur')

async def help_command(update: Update, context):
    await update.message.reply_text('Cobalah mulai dengan "hai", dan tanyakan segala informasi yang ingin kamu dapatkan terkait menu, lokasi, hingga kontak Catering Anggur')

async def error(update: Update, context):
    print(f'Update {update} cause error {context.error}')

# SAVE QUESTION AND ANSWER
def save_interaction_to_json(user_id, question, response):
    filename = 'interactions.json'
    data = {
        "user_id": user_id,
        "question": question,
        "response": response
    }
    
    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            interactions = json.load(file)
    else:
        interactions = []
    
    interactions.append(data)
    
    with open(filename, 'w') as file:
        json.dump(interactions, file, indent=4)

# NLP
def handle_response(sentence):
    sentence = tokenize(sentence)
    sentence = [stem(word) for word in sentence]

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.76:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                text = response.get('text', 'maaf saya tidak mengerti...')
                image = response.get('image', None)
                return {'text': text, 'image': image}
    else:
        return {'text': 'Maaf, saya tidak mengerti...', 'image': None}

# FUNC HANDLE MESSAGE
async def handle_message(update: Update, context):
    start_time = time.time()  # Start time measurement
    
    message_type = update.message.chat.type
    text = update.message.text
    user_id = update.message.chat.id

    print(f'User ({user_id}) in {message_type}: "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text = text.replace(BOT_USERNAME, '').strip()
            response = handle_response(new_text)
            save_interaction_to_json(user_id, new_text, response['text'])
        else:
            return
    else:
        response = handle_response(text)
        save_interaction_to_json(user_id, text, response['text'])

    end_time = time.time()  # End time measurement
    response_time = end_time - start_time  # Calculate the response time
    print('Response time:', response_time, 'seconds')  # Print response time

    print('Bot:', response['text'])
    await update.message.reply_text(f"{response['text']}")

    if response['image']:
        for images in response['image']:
            await update.message.reply_photo(photo=images)

if __name__ == '__main__':
    print('starting...')
    app = Application.builder().token(TOKEN).build()

    # command
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))

    # message
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # error
    app.add_error_handler(error)

    # polling the bot
    print('polling....')
    app.run_polling(poll_interval=3)
