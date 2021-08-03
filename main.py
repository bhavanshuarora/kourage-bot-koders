import os
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import pandas as pd
import numpy as np
import discord
import csv
import pandas as pd
import datetime
from discord.ext import tasks, commands
import logging
import platform
from colorama import init
from termcolor import colored
import time

intents = discord.Intents.all()
intents.reactions = True
intents.members = True
intents.guilds = True
machine = platform.node()
init()

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

class Logger:
    def __init__(self, app):
        self.app = app
    def info(self, message):
        print(colored(f'[{time.asctime(time.localtime())}] [{machine}] [{self.app}] {message}', 'yellow'))
    def success(self, message):
        print(colored(f'[{time.asctime(time.localtime())}] [{machine}] [{self.app}] {message}', 'green'))
    def error(self, message):
        print(colored(f'[{time.asctime(time.localtime())}] [{machine}] [{self.app}] {message}', 'red'))
    def color(self, message, color):
        print(colored(f'[{time.asctime(time.localtime())}] [{machine}] [{self.app}] {message}', color))

logger = Logger("kourage-sentiment")

nltk.download('vader_lexicon')

class MyBot(commands.Bot):
    channels = [ 676375821379829771, 745152122106150932, 848837318244565013, 778608361896935424, 753586853160026143, 859063596331827200 ]
    def __init__(self, command_prefix, self_bot, intent):
        commands.Bot.__init__(self, command_prefix=command_prefix, self_bot=self_bot)

    async def on_ready(self):
        logger.info("We have logged in as {0.user}".format(bot))
        logger.success("~Kourage bot is running at 0.1.0")
        self.message_history.start()
        

    @tasks.loop(hours=24)
    async def message_history(self):
        logger.info("'~message_history' called.")
        data = pd.DataFrame(columns=['content', 'time', 'author'])
        for ch in self.channels:
            channel = self.get_channel(ch)
            print(channel)
            logger.info('~list of channels ' + channel)
            channel.history(limit=10)
            async for msg in channel.history(limit=1000):           # set high limit
                Flag = True    # Trigger to fetch current date messages
                if(msg.created_at.strftime('%Y-%m-%d') == datetime.date.today().strftime('%Y-%m-%d')):
                    data = data.append({'content': msg.content, 'time': msg.created_at, 'author': msg.author.name}, ignore_index=True)
                else:
                    Flag = False  # if date doesn't it means that its the previous day so we break the loop
                if(Flag == False):
                    file_location = "temp.csv"   
                    data.to_csv(file_location, index=False)
        await self.sentimentAnalysis()        
        channel_send = self.get_channel(869127234361897010)  #graph is to be shared in the particular channel
        await channel_send.send(file=discord.File('temp.png'))
        logger.success("'~message_history' executed successfully.")


    async def sentimentAnalysis(self):
        logger.info('~sentimentAnalysis called.')
        data1 = pd.read_csv("temp.csv")
        data1.dropna(inplace = True)
        datetimeobject = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f').strftime('%m-%d-%Y')
        day = data1.time.apply(datetimeobject)
        data1.time = day
        # Remove URL
        rem_url = lambda x : re.sub(r'^http\S+','',x)  
        data1_url = data1.content.apply(rem_url)
        # Remove puntuations
        remove_puntuations = lambda x: re.sub(r'[^\w\s]','',x)
        data1_url_rp = data1_url.apply(remove_puntuations)
        # Removing stopwords
        stop_words = set(stopwords.words('english'))
        rem_stopwards = lambda x: ' '.join(word.lower() for word in x.split() if word not in stop_words)
        data1_url_rp_sw = data1_url_rp.apply(rem_stopwards)
        rem_extras = lambda x: None if x=='' else x
        data1_url_rp_sw_re = data1_url_rp_sw.apply(rem_extras)
        data1.loc[:, ("content")] = data1_url_rp_sw_re
        data1.dropna(inplace=True)
        data1.reset_index(drop=True,inplace=True)
        sentiment = SentimentIntensityAnalyzer()
        sent = lambda x: sentiment.polarity_scores(x)
        sentiment_score = data1.content.apply(sent)
        score = pd.DataFrame(list(sentiment_score))
        analysis = lambda x: 'neutral' if x==0 else ('positive' if x>0 else 'negative')
        data1['compound'] = score.compound.apply(analysis)
        df = data1.groupby(['author','compound']).size().unstack(fill_value=0)
        df1 = df.iloc[:, 0:].apply(lambda x: x.div(x.sum()).mul(100), axis=1).astype(int)
        df1.plot(kind='bar')
        plt.figure(figsize = (20, 10))
        plt.xticks(range(len(df1.index)),df1.index,rotation=90)    #,fontsize =3
        plt.xlabel("Names")
        plt.ylabel("Percentage")
        plt.tight_layout()
        plt.savefig('temp.png')
        plt.clf()
        logger.success("'~Sentiment analysis executed successfully.")
        return
bot = MyBot(command_prefix='~', self_bot=False, intent=intents)
bot.run(os.environ.get('TOKEN'))

