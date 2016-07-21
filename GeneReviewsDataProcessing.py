import json

with open('disease_author_omimID_dict.json', 'r') as f:
    full_dict = json.load(f)
    
#a.replace('and ', '').strip('.').split(', ')

for i in full_dict:
    if ',' in full_dict[i]['authors']:
        full_dict[i]['authors'] = full_dict[i]['authors'].replace('and ', '').strip('.').split(', ')
    else:
        full_dict[i]['authors'] = full_dict[i]['authors'].replace(' and', ',').strip('.').split(', ')
    
