from urllib.request import urlopen
import json
import logging
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def omim_page_scraping(start_num, limit):
    '''
    This function is to scrape one page of omim website by using omim api.
    
    input: start_num: the index of omim id start
           limit: number of omim id references should be shown on one page
    output: json_data: a dictionary with all data in one omim page in it
    
    '''
    #The limit cannot exceed 20
    start_url = "http://api.omim.org/api/entry/search?search=*&filter=&fields=&retrieve=&start=" + str(start_num) + "&limit=" + str(limit) + "20&sort=&operator=&include=referenceList&format=json&apiKey=V96K5rHxTuKYhWIdfAx-IQ"
    data = urlopen(start_url)
    json_data = json.loads(data.read().decode('utf-8'))
    
    return json_data


def get_total_results():
    json_data = omim_page_scraping(1, 20)
    total_results = json_data['omim']['searchResponse']['totalResults']    
    return total_results


def omim_reference_scrape_full():
    
    #The code below is used to scrape data from the OMIM website by using its API
    #It stores entry list into a file and stored in the computer    
    total_results = get_total_results()
    entry_list = []
    new_entry_list = []  
    for i in range(1240):
        json_data = omim_page_scraping(i*20, 20)
        entry_list = entry_list + json_data['omim']['searchResponse']['entryList']
        print('count: ' + str(i))
    
    for entry in entry_list:
        if entry['entry']['status'] == 'live':
            new_entry_list.append(entry)    
    
    with open('omim_entry_set.json', 'w') as file:
        json.dump(entry_set, file)    
    
    with open('new_entry_list.json', 'w') as file:
        json.dump(new_entry_list, file)
        
        
def rearrange_omim_info():
    
    '''
    This function is used to rearrange the data in the omim dictionary.
    The data in the new dictionary ('omim_full_dict.json') has a format of
    
    OMIMid/
         - title   (the name of the disease)
         - pubList/
           - pubmedID/
             - journal
             - title
             - authors
    
    This dictionary is stored in the computer directly.
    
    input: None
    output: None
                
    '''
    
    omim_dict = {}
    count = 0   #count number of articles not in Pubmed
    count1 = 1
    count2 = 0  #count number of articles in Pubmed
    
    with open('new_entry_list.json', 'r') as file:
        entry_list = json.load(file)
        
    
    for entry in entry_list:
        omim_dict[entry['entry']['mimNumber']] = {}
        omim_dict[entry['entry']['mimNumber']]['title'] = entry['entry']['titles']['preferredTitle']
        print('count entry: ' + str(count1))
        pub_dict = {}
        count1 += 1
        for pub in entry['entry']['referenceList']:
            if 'pubmedID' in pub['reference'].keys():
                #count2 += 1
                pub_dict[pub['reference']['pubmedID']] = {'journal': pub['reference']['source'], 'title': pub['reference']['title'], 'authors': pub['reference']['authors']}
                pub_dict[pub['reference']['pubmedID']]['authors'] = pub_dict[pub['reference']['pubmedID']]['authors'].split(', ')
            #else:
                #count += 1
        omim_dict[entry['entry']['mimNumber']]['pubList'] = pub_dict
    
    with open('omim_full_dict.json', 'w') as new_file:
        json.dump(omim_dict, new_file)
    
    return

def journal_date_parse(string):
    
    source_re = re.compile(r'((?:[a-zA-Z]+\s+)+).+(\d{4}).*')
    match = source_re.match(string)
    journal_list = []
    if match:
        journal = match.group(1).strip()
        date = int(match.group(2))
        journal_list.append(journal)
        journal_list.append(date)
    return journal_list

if __name__ == '__main__':
    
    
    with open('omim_full_dict.json', 'r') as file:
        omim_dict = json.load(file)
        
    journal_list = []
    date_list = []
    
    for omim in omim_dict:
        for pub in omim_dict[omim]['pubList']:
            journal = []
            journal = journal_date_parse(omim_dict[omim]['pubList'][pub]['journal'])
            if journal:
                journal_list.append(journal[0])
                date_list.append(journal[1])
                
    journal_distribution = Counter(journal_list)
    date_distribution = Counter(date_list)
    
    date_array = []
    for date in date_distribution:
        date_array.append(date)
    
    date_array = sorted(date_array)
    date = np.array(date_array)
    
    suffix_list = ['Jr.', 'I', 'II', 'III']
    
    count = 0
    count1 = 0
    
    
    #Testing code
    for omim in omim_dict:
        for pub in omim_dict[omim]['pubList']:
            count1 += 1
            #print(len(omim_dict[omim]['pubList'][pub]['authors']))
            if len(omim_dict[omim]['pubList'][pub]['authors']) % 2 != 0:
                if 'Jr.' not in omim_dict[omim]['pubList'][pub]['authors']:
                    if 'III' not in omim_dict[omim]['pubList'][pub]['authors']:
                        if 'II' not in omim_dict[omim]['pubList'][pub]['authors']:
                            if 'IV' not in omim_dict[omim]['pubList'][pub]['authors']:
                                if len(omim_dict[omim]['pubList'][pub]['authors']) != 1:
                                    flag = 0
                                    for element in omim_dict[omim]['pubList'][pub]['authors']:
                                        if '{' in element:
                                            flag = 1
                                            break
                                    if not flag:
                                        count += 1
                                        print(omim_dict[omim]['pubList'][pub]['authors'])    
    
    
    
    
    
                

        
    
    


        
    

    
    
        

