import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import math
from collections import Counter
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import xgboost as xgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import os
import socket
import whois
from datetime import datetime
import time
from bs4 import BeautifulSoup
import urllib
import bs4
import os
def string_input(data_str):
    import pandas as pd
    lines = data_str.split('\n')
    df = pd.DataFrame(lines, columns=['URL'])
    import re
    def having_ip_address(url):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
        if match:
            return 1
        else:
            return 0
    df['use_of_ip'] = df['URL'].apply(lambda i: having_ip_address(i))
    from urllib.parse import urlparse

    def abnormal_url(url):
        hostname = urlparse(url).hostname
        hostname = str(hostname)
        escaped_hostname = re.escape(hostname)
        match = re.search(escaped_hostname, url)
        if match:
            return 1
        else:
            return 0
    df['abnormal_url'] = df['URL'].apply(lambda i: abnormal_url(i))
    df['count.'] = df['URL'].apply(lambda i: i.count('.'))
    df['count-www'] = df['URL'].apply(lambda i: i.count('www'))
    df['count@'] = df['URL'].apply(lambda i: i.count('@'))
    from urllib.parse import urlparse
    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')
    df['count_dir'] = df['URL'].apply(lambda i: no_of_dir(i))
    def no_of_embed(url):
        urldir = urlparse(url).path
        return urldir.count('//')
    df['count_embed_domian'] = df['URL'].apply(lambda i: no_of_embed(i))
    def shortening_service(url):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                        'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                        'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                        'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                        'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                        'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                        'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                        'tr\.im|link\.zip\.net',
                        url)
        if match:
            return 1
        else:
            return 0
    df['short_url'] = df['URL'].apply(lambda i: shortening_service(i))
    df['count-https'] = df['URL'].apply(lambda i : i.count('https'))
    df['count-http'] = df['URL'].apply(lambda i : i.count('http'))
    df['count%'] = df['URL'].apply(lambda i: i.count('%'))
    df['count?'] = df['URL'].apply(lambda i: i.count('?'))
    df['count-'] = df['URL'].apply(lambda i: i.count('-'))
    df['count='] = df['URL'].apply(lambda i: i.count('='))
    df['url_length'] = df['URL'].apply(lambda i: len(str(i)))
    df['hostname_length'] = df['URL'].apply(lambda i: len(urlparse(i).netloc))
    def suspicious_words(url):
        match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                        url)
        if match:
            return 1
        else:
            return 0
    df['sus_url'] = df['URL'].apply(lambda i: suspicious_words(i))
    from urllib.parse import urlparse
    from tld import get_tld
    import os.path
    def fd_length(url):
        urlpath= urlparse(url).path
        try:
            return len(urlpath.split('/')[1])
        except:
            return 0

    df['fd_length'] = df['URL'].apply(lambda i: fd_length(i))
    df['tld'] = df['URL'].apply(lambda i: get_tld(i,fail_silently=True))
    def tld_length(tld):
        try:
            return len(tld)
        except:
            return -1

    df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))
    def digit_count(url):
        digits = 0
        for i in url:
            if i.isnumeric():
                digits = digits + 1
        return digits
    df['count-digits']= df['URL'].apply(lambda i: digit_count(i))
    df = df.drop(labels="tld", axis=1)
    df = df.drop("URL", axis=1)
    X = df[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
        'count_dir', 'count_embed_domian', 'short_url', 'count-https',
        'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
        'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',]]
    import pickle
    with open("gbdt_model.pkl", "rb") as f:
        model = pickle.load(f)
    y_pred1 = model.predict(X)
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    y_pred2 = model.predict(X)
    with open("lgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    y_pred3 = model.predict(X)
    y_pred=(y_pred1[0].astype(int))+(y_pred2[0].astype(int))+(y_pred3[0].astype(int))
    y_pred=y_pred/3
    if(y_pred>0.5):
        return "Malicious"
    else:
        return "Benign"
