import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn import tree
from urllib.error import HTTPError
import bs4
import requests


pd.options.display.width = None
pd.options.display.max_columns = None

#приведение баз данных к рабочему виду
def dfs_prep():

    scores = pd.read_csv('/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/оценки рейтеров/all_nancy+dany.csv', delimiter=';', header=0, index_col='Student_Id')
    actions = pd.read_csv('/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/Автоматизация/CT_Privacy_Scoring.xlsm - Privacy_data.csv', header=1, index_col="Login")

    scores_links = scores[['arg1_src_rel', 'arg2_src_rel', 'arg3_src_rel', 'arg4_src_rel', 'arg5_src_rel', 'arg6_src_rel',
                           'arg7_src_rel', 'arg8_src_rel', 'arg9_src_rel', 'arg10_src_rel', 'arg11_src_rel',
                           'arg12_src_rel', 'arg13_src_rel', 'arg14_src_rel', 'essay_src1_rel', 'essay_src2_rel']]
    actions_links = actions[['CD2.1_Essay_Arguments__LinkText1', 'CD2.1_Essay_Arguments__LinkText2', 'CD2.1_Essay_Arguments__LinkText3',
                             'CD2.1_Essay_Arguments__LinkText4', 'CD2.1_Essay_Arguments__LinkText5', 'CD2.1_Essay_Arguments__LinkText6',
                             'CD2.1_Essay_Arguments__LinkText7', 'CD2.1_Essay_Arguments__LinkText8', 'CD2.1_Essay_Arguments__LinkText9',
                             'CD2.1_Essay_Arguments__LinkText10', 'CD2.1_Essay_Arguments__LinkText11', 'CD2.1_Essay_Arguments__LinkText12',
                             'CD2.1_Essay_Arguments__LinkText13', 'CD2.1_Essay_Arguments__LinkText14', 'CD2.1_Essay_EssayLinks__LinkText1',
                             'CD2.1_Essay_EssayLinks__LinkText2']]

    df_links = pd.DataFrame()
    scores = []
    links = []
    for index, row in scores_links.iterrows():
        try:
            score = row.T.dropna().tolist()
            link = actions_links.loc[index].T.dropna().values.tolist()
            if len(score) == len(link):
                scores.extend(row.T.dropna().tolist())
                links.extend(actions_links.loc[index].T.dropna().values.tolist())
        except KeyError:
            pass
    df_links['scores'] = scores
    df_links['links'] = links

    return df_links

df_links = dfs_prep()

# скачивание 500 слов со страницы веб-сайта
def beautiful_soup(df_links):
    pattern_not_del = "^(http|https)://"
    filter = df_links.links.str.contains(pattern_not_del, case=False)
    link_text = pd.DataFrame(df_links[filter])
    link_text['text'] = np.NaN
    for index, row in link_text.iterrows():
        try:
            response = requests.get(link_text['links'].loc[index])
            response.raise_for_status()
        except HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')  # Python 3.6
        except Exception as err:
            print(f'Other error occurred: {err}')  # Python 3.6
        else:
            soup = bs4.BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            lines = (line.strip("\n").replace(u'\xa0', ' ') for line in text.splitlines())
            chunks = [phrase.strip() for line in lines for phrase in line.split("  ")]
            text = ''.join(chunk for chunk in chunks[1000:1500] if chunk)
            rus = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890 ')
            conv_text = [symbol for symbol in text if symbol.lower() in rus]
            link_text['text'].loc[index] = ''.join(conv_text)
            link_text.dropna(inplace=True)
    link_text.to_csv('/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/Автоматизация/df_links.csv')
    return link_text
df_links = pd.read_csv('/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/Автоматизация/df_links.csv')
df_links.dropna(inplace=True)
#df_links = beautiful_soup(df_links)

#разбиение базы запросов на тест и трейн
def split_links(df_links):
    x_train, x_test, y_train, y_test = train_test_split(df_links[['text']], df_links['scores'], stratify=df_links['scores'], test_size=0.31)
    return x_train, x_test, y_train, y_test

#запуск модели Случайного Леса для базы запросов
def links_forest():
    x_train, x_test, y_train, y_test = split_links(df_links)

    #выбор параметров модели
    parameters = {'forest__n_estimators': [int(x) for x in np.linspace(start=10, stop=400, num=100)],
        'forest__max_depth': [int(x) for x in np.linspace(24, 28, num=2)]}

    #предобработка данных для модели
    my_stop_words = stopwords.words('russian')
    column_trans = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(analyzer=u'word',
                                        stop_words=list(my_stop_words),
                                        encoding='utf-8',
                                        max_df=60,
                                        ngram_range=(1, 1)),
             'text')])

    # добавление классификатора Случайного Леса
    pipeline = Pipeline([
        ('features', column_trans),
        ('forest', RandomForestClassifier(class_weight="balanced", random_state=12345))
    ])

    # Поиск наилучшей модели
    rf_model = GridSearchCV(pipeline,
                               param_grid=parameters,
                               n_jobs=-1,
                               error_score='raise',
                               scoring='f1_weighted',
                               return_train_score=True,
                               verbose=1)

    #Фит и принты лучших параметров
    rf_model.fit(x_train, y_train)
    best = rf_model.best_estimator_
    print(rf_model.best_estimator_)
    print(rf_model.best_score_)

    # Принты метрик качества
    test_pred = best.predict(x_test)
    print(classification_report(y_test, test_pred, digits=3))

links_forest()





