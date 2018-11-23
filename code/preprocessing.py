import pandas as pd


for i in range(12,22):

    df = pd.read_csv('fire-2018-11-{}.csv'.format(i))

    df['tweet'] = df['tweet'].str.replace(r"http\S+", "")
    df['tweet'] = df['tweet'].str.replace(r"\n+", "")


    df.to_csv('fire-2018-11-{}_new.csv'.format(i), index=False)