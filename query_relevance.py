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


pd.options.display.width = None
pd.options.display.max_columns = None

#приведение баз данных к рабочему виду
def dfs_prep():

    scores = pd.read_csv('/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/оценки рейтеров/all_nancy+dany.csv', delimiter=';', header=0, index_col='Student_Id')
    actions = pd.read_csv('/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/Автоматизация/CT_Privacy_Scoring.xlsm - Privacy_data.csv', header=1, index_col="Login")

    scores_query = scores[['query1_query_rel', 'query2_query_rel', 'query3_query_rel', 'query4_query_rel',
                           'query5_query_rel', 'query6_query_rel']]
    actions_query = actions[['CD2.1_Essay_Requests__RequestText1', 'CD2.1_Essay_Requests__RequestText2',
                             'CD2.1_Essay_Requests__RequestText3', 'CD2.1_Essay_Requests__RequestText4',
                             'CD2.1_Essay_Requests__RequestText5', 'CD2.1_Essay_Requests__RequestText6']]

    df_query = pd.DataFrame()
    scores = []
    query = []
    for index, row in scores_query.iterrows():
            try:
                score = row.T.dropna().tolist()
                q = actions_query.loc[index].T.dropna().values.tolist()
                if len(score) == len(q):
                    scores.extend(row.T.dropna().tolist())
                    query.extend(actions_query.loc[index].T.dropna().values.tolist())
            except KeyError:
                pass
    df_query['scores'] = scores
    df_query['query'] = query
    length = []
    for index, row in df_query.iterrows():
        number = len(row['query'].split(' '))
        length.append(number)
    df_query['length'] = length

    return df_query

df_query = dfs_prep()

#разбиение базы запросов на тест и трейн
def split_query(df_query):
    x_train, x_test, y_train, y_test = train_test_split(df_query[['query', 'length']], df_query['scores'], stratify=df_query['scores'], test_size=0.30)
    return x_train, x_test, y_train, y_test

#запуск модели Случайного Леса для базы запросов
def query_forest():
    x_train, x_test, y_train, y_test = split_query(df_query)

    #выбор параметров модели
    parameters = {'forest__n_estimators': [int(x) for x in np.linspace(start=10, stop=400, num=100)],
        'forest__max_depth': [int(x) for x in np.linspace(24, 28, num=2)]}

    #предобработка данных для модели
    numeric_features = ["length"]
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())])
    my_stop_words = stopwords.words('russian')
    column_trans = ColumnTransformer(
        transformers=[
            ('query', TfidfVectorizer(analyzer=u'word',
                                        stop_words=list(my_stop_words),
                                        encoding='utf-8',
                                        max_df=60,
                                        ngram_range=(1,1)),
             'query'),
            ("num",
             numeric_transformer,
             numeric_features)])

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

query_forest()





