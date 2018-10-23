from xml.dom import minidom
import json
import xml.etree.ElementTree as ET
import time
import datetime
import pprint

ARTICLE_DATA = "/PATH/..."
GROUND_TRUTH_DATA = ""

ARTICLE_AS_KEY = "article"
ID_AS_KEY = "id"
HYPERPARTISAN_AS_KEY = "hyperpartisan"

FIRST_ELEMENT_AS_ID = 1

# get labels
labels_xml = minidom.parse(GROUND_TRUTH_DATA)

article_nodes_with_labels = labels_xml.getElementsByTagName(ARTICLE_AS_KEY)
article_label_map = dict()

for article_label in article_nodes_with_labels:
    article_label_map[article_label.attributes[ID_AS_KEY].value]= article_label.attributes[HYPERPARTISAN_AS_KEY].value

articles_json = dict()
articles_json[ARTICLE_AS_KEY] = list()

for event, element in ET.iterparse(ARTICLE_DATA):
    if element.tag == "article":
        counter = 0
        article_id, article_date, article_title = None, None, None
        for item_tuple in element.items():

            item = item_tuple[FIRST_ELEMENT_AS_ID]

            if counter == 0:
                article_id = item
            elif counter == 1:
                article_date = item
            elif counter == 2:
                article_title = item

            counter += 1

        article_text = ET.tostring(element, method='text').decode().strip()
        article_text = article_text.replace("\n","")
        current_article_json = dict()
        current_article_json["id"] = article_id
        current_article_json["published-at"] = article_date
        current_article_json["title"] = article_title
        current_article_json["text"] = article_text
        current_article_json["hyperpartisan"] = article_label_map[article_id]

        articles_json["articles"].append(current_article_json)

        if len(articles_json["articles"])%50000 == 0:
            print("{} articles in dict".format(len(articles_json["articles"])))

        element.clear()

json_file_name =  ARTICLE_DATA.split("/")[len(ARTICLE_DATA.split("/"))-1] + ".json"
with open(json_file_name, 'w', encoding='utf-8') as outputf:
    json.dump(articles_json["articles"], outputf, indent=3)
