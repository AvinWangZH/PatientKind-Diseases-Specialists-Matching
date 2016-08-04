"""
What does this file do?
"""

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pickle
import json
import re
import logging


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

def build_gene_reviews_url_dict():
    '''
    Introduction: This function is used to scrape the gene reveiws web page to 
    get all links of diseases which can be used to further website crawl in next 
    step.
    
    Input: None
    
    Output: url_dict (dictionary): keys are urls and values are disease names
    '''
    
    
    #initialize variables which will be used in this program
    url_dict = {}      #key: links| value: disease names
    
    #get and parse the html of gene reviews website by using beautiful soup
    url = 'http://www.ncbi.nlm.nih.gov/books/NBK1116/'
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    bsObj = BeautifulSoup(urlopen(req).read(), 'lxml')
    
    
    elements = bsObj.findAll('a', {'class': 'toc-item'})
    
    logging.info('Found {} articles in gene reviews'.format(654))
    
    #put all disease name and link into url_dict
    for ele in elements:
        url_dict[ele.attrs['href'].replace('\n', '')] = ele.text.replace('\n', '')
        count += 1
        if count == total_num_disease:
            break
    
    return url_dict

def get_single_disease_omim_mapping(url):
    '''
    Introduction: this code is to map a signle gene reveiws disease to its related 
    omim Ids.
    
    Input: url (string)
    
    Output: omim_ids --- a list with all related omim ID in it
    '''
    
    url = 'http://www.ncbi.nlm.nih.gov/' + url
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    bsObj = BeautifulSoup(urlopen(req).read(), 'lxml')

    omim_ids = bsObj.findAll('a', {'href': re.compile('/omim/')})  
    
    return omim_ids

def build_gene_reviews_disease_omimid_dict(url_dict):
    '''
    Introduction: this function builds the gene_reviews_disease_omimid_dict which is the 
    dictionary to store all gene reviews diseases and omim id mappings
    
    Input: url_dict --- a dictionary stores all urls names of gene reviews diseases 
    
    Output: gene_reviews_disease_omimid_dict --- a dictionary which contains all gene reviews diseases
    and omim Id mappings
    
    '''    
    
    #Initialize gene_reviews_disease_omimid_dict to store gene reveiws and omim id data
    gene_reviews_disease_omimid_dict = {}
    
    #retrieve each disease in the dictionary of gene reviews
    for i in url_dict:
        
        #print(gene_reviews_disease_omimid_dict[url_dict[i]])
        
        id_list = []  #list of omim ids for the disease
        omim_id = get_single_disease_omim_mapping(i)
        
        
        #check if a disease has a related omim id or not and fliter the unrelated data
        if omim_id != []:
            for j in omim_id:
                print(j.text)
                if j.text != 'View All in omim':
                    id_list.append(j.text)
        
        #put each list into gene_reviews_disease_omimid_dict
        disease_name = url_dict[i]
        gene_reviews_disease_omimid_dict[disease_name] = id_list
    
    return gene_reviews_disease_omimid_dict


def parse_args(args):
    from argparse import ArgumentParser
    description = __doc__.strip()
    
    parser = ArgumentParser(description=description)
    parser.add_argument('urls_filename')
    parser.add_argument('disease_omim_map_filename')

    return parser.parse_args(args)


def fetch_urls(urls_filename):
    if os.path.isfile(urls_filename):
        logging.warning('File already exists: {}\nLoading previously scraped URLs...'.format(urls_filename))
        url_dict = read_json_file(urls_filename)
    else:
        logging.info('Scraping URLs of diseases from gene reviews website...')
        url_dict = build_gene_reviews_url_dict()
        
        logging.info('Saving scraped data to file: {}'.format(urls_filename))
        write_json_file(urls_filename, url_dict)
    
    return url_dict


def fetch_disease_omim_map(url_dict, disease_omim_map_filename):
    if os.path.isfile(disease_omim_map_filename):
        logging.warning('File already exists: {}\nLoading previous omim mappings...'.format(disease_omim_map_filename))
        gene_reviews_disease_omimid_dict = load_json_file(disease_omim_map_filename)
    else:
        logging.info('Parsing omim IDs from gene reviews pages...')
        gene_reviews_disease_omimid_dict = build_gene_reviews_disease_omimid_dict(url_dict)
        
        logging.info('Saving disease-omim mapping to file: {}'.format(disease_omim_map_filename))
        write_json_file(disease_omim_map_filename, gene_reviews_disease_omimid_dict)
        
    return gene_reviews_disease_omimid_dict


def main(urls_filename, disease_omim_map_filename):
    url_dict = load_urls(urls_filename)
    gene_reviews_disease_omimid_dict = fetch_disease_omim_map(url_dict, disease_omim_map_filename)
    

if __name__ == '__main__':
    args = parse_args(args)
    logging.basicConfig(level='INFO')
    main(**vars(args))
    




    

    

    

