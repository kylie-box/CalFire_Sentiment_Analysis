import pandas as pd
from nltk.tokenize.casual import TweetTokenizer
import unicodedata
from unidecode import unidecode


INPUT_FILE = './data/data_preprocessed/automatic_sentiment_tag_by_emoji/fire-2018-11-{}_emoji_tagged.csv'
TAG_FILE = './data/first_50_tagged/{}-00-50.csv'


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


for i in range(13,19):
    df_emoji = pd.read_csv(INPUT_FILE.format(str(i)), delimiter=',')
    df_human = pd.read_csv(TAG_FILE.format(str(i)), delimiter=',', encoding="ISO-8859-1")
    df_emoji['human_score'] = df_human['tag']

    # tknzr = TweetTokenizer()

    df_emoji['tweet'] = df_emoji['tweet'].apply(deEmojify, convert_dtype=False)
    df_emoji.to_csv(INPUT_FILE.format(str(i)),index=False)

# df_emoji['tweet'] = df_emoji['tweet'].apply(tknzr.tokenize, convert_dtype=False)

