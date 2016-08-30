from urllib.parse import urlparse
from copy import deepcopy
import urllib.request
import pickle
import json

def parse_author_name(filename):
    #Open a file and load the data
    with open(filename, encoding = 'utf-8') as f:
        data = f.readlines()
    

    for i in range(len(data)):
        #delete the last character which is a '.'
        data[i] = data[i][0:len(data[i])-2]
    
        #if there is an 'and' in the name list, we substitute ','
        #it can make the data into the same format which means that all author names 
        #are separated by ','
        if(data[i].find('and') != -1):
            if(data[i][data[i].find(' and')-1] == ','):
                data[i] = data[i].replace(' and', '')   
            else:
                data[i] = data[i].replace(' and', ',')  
            
        #separate the string into a list of author names
        data[i] = data[i].split(', ')
        
    return data
        
        
def diseaseList(filename):
    #Open a file and load the data
    with open(filename, encoding = 'utf-8') as f:
        data = f.readlines()    
    
    for i in range(len(data)):
        if data[i].find(' \n') != -1:
            data[i] = data[i].replace(' \n', '')
        else:
            data[i] = data[i].replace('\n', '')
    return data

def publicationIdList(author_name):
    #Input: any author's name
    #Output: the Id of publiscations of that author
    
    #intialize variable for storing the publication ID
    pub_list = []
    
    #open the url to get the number of publications as Count
    opener = urllib.request.FancyURLopener({})
    url = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=" + author_name.replace(' ', '+')  
    f1 = opener.open(url)
    content_initial = str(f1.read())
    
    Count_num_start = content_initial.find('<Count>')
    Count_num_end = content_initial.find('</Count>')
    Count = content_initial[Count_num_start + 7:Count_num_end]    
    
    #change the url to have all publications in one page and open the new url
    opener = urllib.request.FancyURLopener({})
    url = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=" + author_name.replace(' ', '+') + '&RetMax=' + Count
    f2 = opener.open(url)
    content_with_all_pub = str(f2.read())
    
    #store the parsed data in temp
    temp = content_with_all_pub.split('<Id>')
    
    #put all publication ID into a list
    for i in range(len(temp)):
        if(temp[i][8:13] == '</Id>'):
            pub_list.append(temp[i][0:8])
    
    #return the publication list
    return pub_list
    
#to check a str is in ascii or not
def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def get_paper_url_byID(id_list):
    all_ids = ''
    sum = 0
    for i in range(len(id_list)):
        all_ids += (',%' + id_list[i])

    all_ids = all_ids[2:len(all_ids)-1]
    url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=' + all_ids
    return url
    
def scrape_all_author_pmid(author_list):
    
    author_list_encode = deepcopy(author_list)
    for i in range(len(author_list)):
        for j in range(len(author_list[i])):
            author_list_encode[i][j] = urllib.parse.quote(author_list[i][j])
                   
    
    #make a copy of author_list to make the new list has the same size of author_list
    pub_id_list = deepcopy(author_list_encode)
    
    with open('data.txt', 'r') as outfile:
        data = json.load(outfile) 
        
    url = get_paper_url_byID(data)

    #scrape data online and put the data into the new list: all author publication IDs
    for i in range(len(author_list_encode)):
        for j in range(len(author_list_encode[i])):
            
            author_name = author_list_encode[i][j]
            pub_id_list_personal = publicationIdList(author_name)
            
            pub_id_list[i][j] = pub_id_list_personal    


if __name__ == '__main__':
    #to get authors list and the diseases list corresponding to it
    info = pickle.load(open('author_info.p', 'rb'))
    author_list = parse_author_name('Author Names.txt')
    disease_list = diseaseList('Diseases.txt')
    
    author_info = {}
    
    for i in range(len(author_list)):
        for j in range(len(author_list[i])):
            if author_list[i][j] not in author_info:
                author_info[author_list[i][j]] = {}
            if 'Disease' not in author_info[author_list[i][j]]:
                author_info[author_list[i][j]]['Disease'] = []
            author_info[author_list[i][j]]['Disease'].append(disease_list[i])
            if 'Publications' not in author_info[author_list[i][j]]:
                author_info[author_list[i][j]]['Publications'] = info[author_list[i][j]]['Publications']

    pickle.dump(author_info, open('author_info.p', 'wb'))
    
    opener = urllib.request.FancyURLopener({})
    url = get_paper_url_byID(author_info['Agatino Battaglia']['Publications'])
    f2 = opener.open(url)
    content = str(f2.read())    
