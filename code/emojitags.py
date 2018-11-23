import pandas as pd

# with open('emoji_dict.txt','r') as f:
#     for line in f:
#         print(line.split())

import unicodedata
from unidecode import unidecode
import re

df = pd.read_csv('fire-2018-11-21_new.csv')

def deEmojify(inputString):
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
                    returnString += " [x]"

    return returnString



emoji_dict = {}


with open('emoji_dict.txt',"r") as f:
    for line in f:
        name = deEmojify(line).strip()[:-2].strip()
        emoji_dict.update({name:
                           deEmojify(line).strip()[-2:].strip()})

for k in list(emoji_dict.keys()):
    if k.startswith('[x]') or k.startswith('[?]') or k.endswith('[x]') or k.endswith('[?]'):
        emoji_dict.pop(k)

for row in df['tweet']:
    row = deEmojify(row)
    key = row[row.find("[")+1:row.find("]")]
    print(key)
# s = deEmojify(s)
# s = s[s.find("[")+1:s.find("]")]
# print(s)
