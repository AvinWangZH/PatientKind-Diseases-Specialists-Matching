from scipy.sparse import coo_matrix
from urllib.request import urlopen
from collections import Counter
import numpy as np
import logging
import pickle
import json
import re
import os

def omim_page_scraping(start_num, num_each_page):
    '''
    This function is to scrape one page of omim website by using omim api.
    
    input: start_num: the index of omim id start
           limit: number of omim id references should be shown on one page
    output: json_data: a dictionary with all data in one omim page in it
    
    '''
    #The limit cannot exceed 20
    start_url = "http://api.omim.org/api/entry/search?search=*&filter=&fields=&retrieve=&start=" + str(start_num) + "&limit=" + str(num_each_page) + "&sort=&operator=&include=referenceList&format=json&apiKey=V96K5rHxTuKYhWIdfAx-IQ"
    data = urlopen(start_url)
    json_data = json.loads(data.read().decode('utf-8'))
    
    return json_data


def get_total_results():
    '''
    "total results" is the total number of omim_id on OMIM website
    '''
    #hard code by looking for the first page of OMIM
    num_each_page = 20
    json_data = omim_page_scraping(1, num_each_page)
    total_results = json_data['omim']['searchResponse']['totalResults']    
    return total_results


def omim_reference_scrape_full():
    #The code below is used to scrape data from the OMIM website by using its API
    #It stores entry list into a file and stored in the computer    
    total_results = get_total_results()
    entry_list = []
    new_entry_list = []
    num_each_page = 20
    for i in range(int(total_results/num_each_page)):
        json_data = omim_page_scraping(i*num_each_page, num_each_page)
        entry_list = entry_list + json_data['omim']['searchResponse']['entryList']
        #print(json_data['omim']['searchResponse']['entryList'])
        print('count: ' + str(i))
    
    for entry in entry_list:
        if entry['entry']['status'] == 'live':
            new_entry_list.append(entry)    
            
    return new_entry_list
        
def name_parsing(author_name):
    '''
    Parsing the names into proper format: "full last name, first initial. middle initial (optional)"
    '''
    name_list = []
    suffix_list = ['Jr.', 'I', 'II', 'III', 'IV']  
        
    index = 0
    while index < len(author_name)-1:
        if '{' in author_name[index]:
            break
        if author_name[index] not in suffix_list:
            name = author_name[index] + ', ' + author_name[index+1]
            index += 2
            name_list.append(name)
        else:
            name += ', ' + author_name[index] 
            index += 1
            name_list[len(name_list) - 1] = name  
    return name_list
        
def rearrange_omim_info(entry_list):
    
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
    count1 = 1
    
    for entry in entry_list:
        omim_dict[entry['entry']['mimNumber']] = {}
        omim_dict[entry['entry']['mimNumber']]['title'] = entry['entry']['titles']['preferredTitle']
        print('count entry: ' + str(count1))
        pub_dict = {}
        count1 += 1
        for pub in entry['entry']['referenceList']:
            if 'pubmedID' in pub['reference'].keys():
                pub_dict[pub['reference']['pubmedID']] = {'journal': pub['reference']['source'], 'title': pub['reference']['title'], 'authors': pub['reference']['authors']}
                pub_dict[pub['reference']['pubmedID']]['authors'] = pub_dict[pub['reference']['pubmedID']]['authors'].split(', ')
                pub_dict[pub['reference']['pubmedID']]['authors'] = name_parsing(pub_dict[pub['reference']['pubmedID']]['authors'])
        omim_dict[entry['entry']['mimNumber']]['pubList'] = pub_dict
    
    return omim_dict

def journal_date_parse(string):
    '''
    This function is used to parse the journal date form a string.
    The string normally has the format of [JournalName] ##, ###, [Year_of_Publication]
    
    input: a string
    output: a two-element list [journalName, year] 
    '''
    source_re = re.compile(r'((?:[a-zA-Z\s\.]+)+).+(\d{4}).*')
    match = source_re.match(string)
    journal_list = []
    if match:
        journal = match.group(1).strip()
        #print(journal+ '/' + string)
        date = int(match.group(2))
        #print(date)
        journal_list.append(journal)
        journal_list.append(date)
    return journal_list

def get_journal_dist(omim_dict):
    '''
    This function is used to get the districution of journal names.
    input: None
    output: journal_distribution: a Counter object with sorted journal data
    '''
        
    journal_list = []
    for omim in omim_dict:
        for pub in omim_dict[omim]['pubList']:
            journal = []
            journal = journal_date_parse(omim_dict[omim]['pubList'][pub]['journal'])
            if journal:
                journal_list.append(journal[0])
                
    journal_distribution = Counter(journal_list)
    
    return journal_distribution


def get_date_dist(omim_dict):
    '''
    This function is used to get the districution of publication years.
    input: None
    output: date_distribution: a Counter object with sorted year data
    '''
        
    date_list = []
    
    for omim in omim_dict:
        for pub in omim_dict[omim]['pubList']:
            journal = []
            journal = journal_date_parse(omim_dict[omim]['pubList'][pub]['journal'])
            if journal:
                date_list.append(journal[1])
                
    date_distribution = Counter(date_list)
    return date_distribution
    

def get_name_dist(omim_dict):
    
    name_list = []
    suffix_list = ['Jr.', 'I', 'II', 'III', 'IV']  
        
    #with open('abandon_list.json', 'r') as file:
        #abandon_list = json.load(file)
        
    for omim in omim_dict:
        for pub in omim_dict[omim]['pubList']:
            author_name = omim_dict[omim]['pubList'][pub]['authors']
            #if pub not in abandon_list:
            for name in author_name:
                name_list.append(name)
                        
    name_list = Counter(name_list)    
    return name_list  
    
def get_omimID_list(omim_dict):
    omimID_list = set()
    for omimID in omim_dict:
        omimID_list.add(omimID)
    return list(omimID_list)

def get_author_list(omim_dict):
    author_list = []
    for omimID in omim_dict:
        for pubmedID in omim_dict[omimID]['pubList']:
            author_list.extend(omim_dict[omimID]['pubList'][pubmedID]['authors'])
    return list(set(author_list))
    
def build_author_omimID_mat(author_list, disease_list, omim_dict):
    author_omimID_mat = np.zeros((len(author_list), len(disease_list)))
    author_index_lookup = dict([(author, index) for (index, author) in enumerate(author_list)])
     
    for disease_index, disease_id in enumerate(disease_list):
        #print('Disease: ', disease_index)
        for pub in omim_dict[disease_id]['pubList'].values():
            #print(pub)
            #print(disease_id)
            for author in pub['authors']:
                #print(author)
                author_index = author_index_lookup.get(author)
                #print(author_index)
                if author_index is not None:
                    #print("Found")
                    author_omimID_mat[author_index][disease_index] += 1
                
    return author_omimID_mat


def get_author_omimID_mat(omim_dict):
     
    omimID_list = get_omimID_list(omim_dict)
    author_list = get_author_list(omim_dict)
        
    mat = build_author_omimID_mat(author_list, omimID_list, omim_dict)
    
    return mat, author_list, omimID_list


if __name__ == '__main__':
    
    #Step 1: get entry list
    entry_list = omim_reference_scrape_full()
    
    #Step 2: get omim dict
    omim_dict = rearrange_omim_info(entry_list)
    
    #Step 3: get author_omimID_mat, author_list, omimID_list
    author_omimID_mat, author_list, omimID_list = get_author_omimID_mat(omim_dict)
    
    #Step 4: transform data in a form of sparse matrix
    sparse_author_omimID_mat = coo_matrix(author_omimID_mat)
    
    #Step 5: store files for next step use
    with open('omim_dict.json', 'w') as file: 
        json.dump(omim_dict, file)
    
    with open('author_list.json', 'w') as file: 
        json.dump(author_list, file)  
        
    with open('omimID_list.json', 'w') as file: 
        json.dump(omimID_list, file)    
    
    with open('author_omimID_mat_coo.p', 'wb') as file:
        pickle.dump(sparse_author_omimID_mat, file)     