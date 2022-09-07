# %%
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

#%%
# Create a new chat bot named Charlie
chatbot = ChatBot('Charlie')

# trainer = ListTrainer(chatbot)
trainer = ChatterBotCorpusTrainer(chatbot)

# trainer.train('chatterbot.corpus.english')

# %%
response = chatbot.get_response('Thats cool')

print(response)
# %%
