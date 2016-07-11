from urllib.request import urlopen
import json
import logging

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
    
   

if __name__ == '__main__':
    
    omim_dict = {}
    count = 0   #count number of articles not in Pubmed
    count1 = 1
    count2 = 0  #count number of articles in Pubmed
    
    with open('new_entry_list.json', 'r') as file:
        entry_list = json.load(file)
        
    
    for entry in entry_list:
        omim_dict[entry['entry']['mimNumber']] = {}
        print('count entry: ' + str(count1))
        pub_list = []
        count1 += 1
        for pub in entry['entry']['referenceList']:
            if 'pubmedID' in pub['reference'].keys():
                count2 += 1
                pub_list.append(pub['reference']['pubmedID'])
                omim_dict[entry['entry']['mimNumber']]['Publications'] = pub_list
            else:
                count += 1
        
    
    
      
    

        
    

    
    
        

