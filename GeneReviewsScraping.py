from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import json
import re


with open('link_list.json', 'r') as file:
    link_list = json.load(file)
    
omim_id_list = []

count = 0

#end with 600
for i in link_list[601:]:
    id_list = []
    req = Request(i, headers={'User-Agent': 'Mozilla/5.0'})
    bsObj = BeautifulSoup(urlopen(req).read())

    omim_id = bsObj.findAll('a', {'href': re.compile('/omim/')})

    if omim_id != []:
        for i in omim_id:
            print(i.text)
            if i.text != 'View All in OMIM':
                id_list.append(i.text)
    
    omim_id_list.append(id_list)
    

    

    

