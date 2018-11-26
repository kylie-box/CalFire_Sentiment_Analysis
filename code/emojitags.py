import pandas as pd

# with open('emoji_dict.txt','r') as f:
#     for line in f:
#         print(line.split())

import unicodedata
from unidecode import unidecode
import re

INPUT_DATA_PATH = './data/data_preprocessed/fire-2018-11-{}_new.csv'
OUTPUT_DATA_PATH = './data/data_preprocessed/automatic_sentiment_tag_by_emoji/fire-2018-11-{}_new.csv'


def deEmojify(inputString):
    # get all emojis in the form of text in a corpora
    returnString = ""

    for character in inputString:
        try:
            character.encode("ascii")
            returnString += character
        except UnicodeEncodeError:
            replaced = unidecode(str(character))
            if replaced != '':
                returnString += replaced
            else:
                try:
                    # a = unicodedata.name(character).lower()
                    # print(a)
                    returnString += " [" + unicodedata.name(character).lower() + "]"
                except ValueError:
                    pass

    return returnString


emoji_dict = {}

with open('emoji_dict.txt', "r") as f:
    for line in f:
        name = deEmojify(line).strip()[:-2].strip()
        # print(name)
        emoji_dict.update({name:int(deEmojify(line).strip()[-2:].strip())})


for date in range(12,22):

    file_name = INPUT_DATA_PATH.format(date)
    df = pd.read_csv(file_name)

    for k in list(emoji_dict.keys()):  # remove all unknown emojis
        if k.startswith('[?]') or k.endswith('[?]') or k == "" or len(re.findall(r'\[.*?\]', k)) > 1:
            emoji_dict.pop(k)

    # get sentiment based on the emoji dictionary
    ct = 0
    for i, row in enumerate(df['tweet']):

        emojis = deEmojify(row)    # one tweet might have multiple Emojis
        key = re.findall(r'\[.*?\]', emojis)   # return the list of emojis

        ct = ct + 1
        emoji_score = 0
        val = 0
        try:
            for k in key:
                val = emoji_dict.get(k)
                if val is None:
                    val = 0
                emoji_score = emoji_score+int(val)
        except:
            continue
        if emoji_score > 0:
            df.at[i, 'score'] = 1
        elif emoji_score < 0:
            df.at[i, 'score'] = 0
            # print(emoji_score)

    write_file = OUTPUT_DATA_PATH.format(date)
    df.to_csv(write_file,index=False)