import numpy as np
import pandas as pd
import gc
import glob
import os
from urlextract import URLExtract
import requests
from urllib.request import urlopen
import tldextract

extractor = URLExtract()

URL_REDIRECT_LOCATION = 'Location'
TIMEOUT = 2 # in seconds for not responding troll pages
PATH = "/social_bias_data/russian-troll-tweets/"
FILE_SUFFIX = "*.csv"
TROLL_TYPE = "RightTroll" # Change this for different type troll links
TWITTER_MESSAGE_CONTENT = "content"

FILENAMES = glob.glob(os.path.join(PATH, FILE_SUFFIX))

def get_domain_from_url(url):
    meta_information = tldextract.extract(url)
    return meta_information.domain + "." + meta_information.suffix

def get_unshortened_url(url):
    last_url = url
    while True:  # we need that logic because there are cyclic troll posts e.g. t.co/sth -> bit.ly/sth -> real_target
        try:
            location = requests.get(last_url, allow_redirects=False, timeout=TIMEOUT)
        except: # Handles not responding, forbidden, not existing and other types of errors
            break

        if URL_REDIRECT_LOCATION not in location.headers:
            break
        last_url = location.headers[URL_REDIRECT_LOCATION]
    return last_url

for file in FILENAMES:
    print("Preprocessing {}".format(file))

df = pd.concat((pd.read_csv(f) for f in FILENAMES))

print("File shape after concatenation {}".format(df.shape))

trolls = df[ df["account_category"] == TROLL_TYPE]

del df
gc.collect()

list_full_urls = []
list_domains = []

length = len(trolls)

for i in range(length): # TODO set length

    if i % 100 == 0:
       print(i)
    troll_text = trolls.iloc[i][TWITTER_MESSAGE_CONTENT]
    all_urls = []


    if extractor.has_urls(troll_text):
        all_urls = extractor.find_urls(troll_text)

    for url in all_urls:
        try:
            unshortened_url = get_unshortened_url(url)
            url_domain = get_domain_from_url(unshortened_url)
            #print(unshortened_url, url_domain)
            list_full_urls.append(unshortened_url)
            list_domains.append(url_domain)
        except:
            print("broken url ", url)
            continue

troll_url_data = pd.DataFrame({"url" : list_full_urls, "domain" : list_domains})
troll_url_data.to_csv("/output/" + TROLL_TYPE + "_urls.csv", index=False)
