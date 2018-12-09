import tweet_tokenizer
from sklearn import semi_supervised
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

OUTPUT_NBEM_FIILE = True

sw = stopwords.words('english')


INPUT_FILE = 'data/data_preprocessed/automatic_sentiment_tag_by_emoji/fire-2018-11-{}_emoji_tagged.csv'


to_remove = (
    # username
    r""" ((@[A-Za-z0-9]+):)
    |
    ((@[A-Za-z0-9]+))
    """,
    #RT
    r"""(^RT$)
    |
    (^rt$)""",
    # https
    r"(http\S+)"
)


def merge_files(wf,rf,num_range):
    fout = open(wf, 'w')
    fout.close()
    fout = open(wf, "a")
    for num in range(num_range[0], num_range[1]):
        if num != num_range[0]:
            m = open(rf.format(str(num)),encoding="utf8", errors='ignore').readlines()
            for line in m[1:]:
                fout.write(line)
        else:
            for line in open(rf.format(str(num)),encoding="utf8", errors='ignore'):
                fout.write(line)
    fout.close()


def preprocessing(f, train_flag=True):
    df = pd.read_csv(f)

    # remove all quotes to avoid strange behaviors
    df['tweet'] = df['tweet'].str.replace(r'"', r'')

    # if automatic score by emojis is not na and no human tags, then assign the emoji sentiment tag to it
    df.loc[~df['score'].isna()&df['human_score'].isna(),'human_score'] = df.loc[~df['score'].isna()&df['human_score'].isna(),'score']
    df.drop('score',axis=1,inplace=True)

    if not train_flag:
        df = df[~df['human_score'].isna()].reset_index(drop=True)

    # predefined regex to remove: username RT/rt hyperlinks
    rm_re = re.compile(r"""(%s)""" % "|".join(to_remove), re.VERBOSE | re.I | re.UNICODE)
    df['tweet'] = df['tweet'].apply(lambda x: rm_re.sub("", x))

    # remove all tweets that have length < 3  after removing all usernames..
    df['len_of_tweet'] = df['tweet'].str.split().apply(len)
    df = df.loc[df['len_of_tweet']>3]

    # shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def nbem_file_format(train_df,test_df,cv):
    train_df['for_nbem'] = train_df['human_score']+1
    test_features = cv.transform(test_df['tweet'])
    test_df['for_nbem'] = test_df['human_score']+1

    nbem_feature = cv.transform(train_df['tweet'])
    feature_ct=len(list(cv.vocabulary_))
    with open('./nbem_labelled_file.txt','w') as a, open('./nbem_unlabelled_file.txt','w') as b:
        a.write('# 2 {}\n'.format(feature_ct))
        b.write('# 2 {}\n'.format(feature_ct))
        for arow,y in zip(nbem_feature.todense(),train_df['for_nbem']):
            if y>0:
                a.write(str(int(y)) + "\t")
                for ind, i in enumerate(arow.A1):
                    if i!=0:
                        a.write(str(ind+1)+":"+str(i)+" ")
                a.write("\n")
            else:
                b.write(str(int(y)) + "\t")
                for ind, i in enumerate(arow.A1):
                    if i != 0:
                        b.write(str(ind + 1) + ":" + str(i) + " ")
                b.write("\n")

    with open('./nbem_test_file.txt','w') as f:
        f.write("# 2 {}\n".format(feature_ct))
        for arow, y in zip(test_features.todense(),test_df['for_nbem']):
            f.write(str(int(y)) + "\t")
            for ind, i in enumerate(arow.A1):
                if i != 0:
                    f.write(str(ind + 1) + ":" + str(i) + " ")
            f.write("\n")


if __name__ == '__main__':
    # merge training files

    out_train = 'data/training/train.csv'
    merge_files(out_train, INPUT_FILE, (12, 19))

    # merge testing files
    out_test = 'data/training/test.csv'
    merge_files(out_test, INPUT_FILE, (19, 22))

    train_df = preprocessing(out_train, train_flag=True)
    test_df = preprocessing(out_test, train_flag=False)

    train_df['human_score'].fillna(-1, inplace=True)
    # test_df['human_score'].fillna(-1, inplace=True)

    num = 2
    print("========== bow{}==============".format(num))
    cv = CountVectorizer(tokenizer=tweet_tokenizer.Tokenizer(preserve_case=False).tokenize, stop_words=sw, ngram_range=(1, num))
    human_only_training = True
    if human_only_training:

        human_tagged = train_df[train_df['human_score'] != -1]
        train_features = cv.fit_transform(human_tagged['tweet'])
        train_Y = human_tagged['human_score']
    else:
        train_features = cv.fit_transform(train_df['tweet'])
        semi_labeling = semi_supervised.LabelSpreading(kernel=str('rbf'),gamma=6.1)
        train_Y = train_df['human_score']
        semi_labeling.fit(train_features.toarray(), train_Y)

        # train_df['semi_supervised_labelling'] = semi_labeling.transduction_
        # train_Y = train_df['semi_supervised_labelling']

    if OUTPUT_NBEM_FIILE:
        nbem_file_format(train_df,test_df,cv)
    test_Y = test_df['human_score']
    test_features = cv.transform(test_df['tweet'])

    # y = EM.fit(train_features,train_Y,10,1e-4)

    #
    # # print(semi_labeling.score(test_features, test_Y))
    #
    nb0 = BernoulliNB()
    nb0.fit(train_features, train_Y)
    print("Naive Bayes with human only label: " + str(nb0.score(test_features, test_Y)))

    lr0 = LogisticRegression()
    lr0.fit(train_features, train_Y)
    print("Logistic with human only label: " + str(lr0.score(test_features, test_Y)))

    svm_0_rec = []
    svm_rec = []
    nb_rec = []
    lr_rec = []

    for i in range(10,100,10):

        print("======================= C = {} =============\n".format(i))
        svm0 = LinearSVC(C=i)
        svm0.fit(train_features, train_Y)
        print("SVM with human only label: "+str(svm0.score(test_features,test_Y)))
        svm_0_rec.append(svm0.score(test_features,test_Y))

        new_train_features = cv.transform(train_df['tweet'])
        new_train_Y = svm0.predict(new_train_features)


        # test class distribution
        # print(sum(test_Y)/len(test_Y))
        param = {
                  'C':[1,10,20,30,40,50,60,70],
                 }

        svm = GridSearchCV(LinearSVC(),param,cv=5,verbose=0)
        svm.fit(new_train_features, new_train_Y)
        print("SVM with pseudo labels: "+str(svm.score(test_features,test_Y)))
        print("SVM best parameter C: {}".format(svm.best_params_))
        svm_rec.append(svm.score(test_features, test_Y))

        nb = BernoulliNB()
        nb.fit(new_train_features,new_train_Y)  # label the untagged data in the training set, retrain on the whole training set
        print("Naive Bayes with pseudo labels: "+str(nb.score(test_features, test_Y)))
        nb_rec.append(nb.score(test_features, test_Y))

        lr = LogisticRegression()
        lr.fit(new_train_features,new_train_Y)
        print("Logistic Regression with pseudo labels: " + str(lr.score(test_features, test_Y)))
        lr_rec.append(lr.score(test_features, test_Y))

    fig = plt.figure(figsize=(10, 5))
    ax=fig.add_subplot(111)
    ax.set_title('SVM penalty C vs accuracy')
    plt.plot(range(10,100,10),svm_0_rec, 'r-',label='svm without pseudo labels')
    plt.plot(range(10, 100, 10), svm_rec, 'b-',label='svm with pseudo labels')
    plt.plot(range(10, 100, 10), nb_rec, 'g-',label='NB with pseudo labels')
    plt.plot(range(10, 100, 10), lr_rec, 'o-',label='Logistic Regression with pseudo labels')
    plt.plot(range(10, 100, 10),np.ones(9)*nb0.score(test_features, test_Y),'g:',label ='NB without pseudo labels')
    plt.plot(range(10, 100, 10),np.ones(9)*lr0.score(test_features, test_Y),'b:',label='Logistic regression without pseudo labels')
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=2)
    ax.set_ylabel('accuracy')
    ax.set_xlabel('penalty C')
    plt.savefig('svm_penaltyC_vs_accuracy_bow2',bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    print("SVM_human_only best: "+str(max(svm_0_rec)))
    print("SVM best: " + str(max(svm_rec)))
    print("NB best: " + str(max(nb_rec)))
    print("LR best: " + str(max(lr_rec)))
