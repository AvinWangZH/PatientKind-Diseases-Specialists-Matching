from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pickle
import json
import re


def read_json_file(filename, var_name):
    
    with open(filename, 'r') as f:
        var_name = json.load(f)
        
    return var_name

def read_pickle_file(filename, var_name):
    
    with open(filename, 'rb') as f:
        var_name = pickle.load(f)
        
    return var_name

def write_json_file(filename, var):
    
    with open(filename, 'w') as f:
        json.dump(var, f)
        
    return


def write_pickle_file(filename, var):
    
    with open(filename, 'wb') as f:
        pickle.dump(var, f)
        
    return

def build_GR_url_dict():
    '''
    Introduction: This function is used to scrape the gene reveiws web page to 
    get all links of diseases which can be used to further website crawl in next 
    step.
    
    Input: None
    
    Output: url_dict (dictionary): keys are urls and values are disease names
    '''
    
    
    #initialize variables which will be used in this program
    url_dict = {}      #key: links| value: disease names
    author_names = {}
    
    #get and parse the html of gene reviews website by using beautiful soup
    url = 'http://www.ncbi.nlm.nih.gov/books/NBK1116/'
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    bsObj = BeautifulSoup(urlopen(req).read(), 'lxml')
    
    
    #elements = bsObj.findAll('a', {'class': 'toc-item'})
    elements2 = bsObj.findAll('li', {'class':'half_rhythm'})
    
    total_num_disease = int(input('How many diseases are on GeneRiviews: '))
    count = 0
    
    #put all disease name and link into url_dict
    for e in elements2:
        ele = e.find('a', {'class': 'toc-item'})
        url_dict[ele.attrs['href'].replace('\n', '')] = ele.text.replace('\n', '')
        author_name = e.findAll('div', {'class': None})[0].text.replace('.', '')
        author_names[ele.text.replace('\n', '')] = author_name
        count += 1
        if count == total_num_disease:
            break
    
    return url_dict, author_names

def get_single_disease_OMIM_mapping(url):
    '''
    Introduction: this code is to map a signle gene reveiws disease to its related 
    OMIM Ids.
    
    Input: url (string)
    
    Output: omim_ids --- a list with all related OMIM ID in it
    '''
    
    url = 'http://www.ncbi.nlm.nih.gov/' + url
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    bsObj = BeautifulSoup(urlopen(req).read(), 'lxml')

    omim_ids = bsObj.findAll('a', {'href': re.compile('/omim/')})  
    
    return omim_ids

def build_GR_disease_author_OMIMid_dict(url_dict, author_names):
    '''
    Introduction: this function builds the GR_disease_OMIMid_dict which is the 
    dictionary to store all gene reviews diseases and OMIM id mappings
    
    Input: url_dict --- a dictionary stores all urls names of GR diseases 
    
    Output: GR_disease_OMIMid_dict --- a dictionary which contains all GR diseases
    and OMIM Id mappings
    
    '''    
    
    #Initialize GR_disease_OMIMid_dict to store gene reveiws and OMIM id data
    GR_disease_author_OMIMid_dict = {}
    
    #retrieve each disease in the dictionary of GR
    for i in url_dict:
        
        #print(GR_disease_OMIMid_dict[url_dict[i]])
        
        id_list = []  #list of omim ids for the disease
        omim_id = get_single_disease_OMIM_mapping(i)
        
        
        #check if a disease has a related omim id or not and fliter the unrelated data
        if omim_id != []:
            for j in omim_id:
                print(j.text)
                if j.text != 'View All in OMIM':
                    id_list.append(j.text)
        
        #put each list into GR_disease_OMIMid_dict
        disease_name = url_dict[i]
        GR_disease_author_OMIMid_dict[disease_name] = {}
        GR_disease_author_OMIMid_dict[disease_name]['omimID_list'] = id_list
        GR_disease_author_OMIMid_dict[disease_name]['authors'] = author_names[disease_name]
    
    return GR_disease_author_OMIMid_dict


if __name__ == '__main__':
    
    #step 1: scrape the urls of diseases from gene reviews main website
    url_dict, author_names = build_GR_url_dict()
    filename = 'url_dict.json'
    
    #Save the file
    write_json_file(filename, url_dict)
    
    #step 2: create GR_disease_OMIMid_dict and store it 
    GR_disease_author_OMIMid_dict = build_GR_disease_author_OMIMid_dict(url_dict, author_names)
    filename = 'GR_disease_author_OMIMid_dict.json'
    
    #Save the file
    write_json_file(filename, GR_disease_author_OMIMid_dict)