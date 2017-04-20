import json
import string
import re
import sys
import numpy as np

def remove_lower(string):
    '''
    This function removes lower case letters in a name.
    for example: name: Zihan Wang can be written as in an OMIM format which is 
    Wang, Z.. Therefore, the lower letters should be eliminated from the first 
    name. The same thing happens for middle names.
    '''
    new = ''
    remove_low = lambda text: re.sub('[a-z]', '', text)
    new = remove_low(string)
    return new

def transform_names_to_omim_format(full_dict):
    '''
    In GeneReviews, author names are in a format of full name. However, in order
    to make name match with OMIM names, we have to transform the full names into
    format of "Full last name, First Initial. Middle Initial". This function 
    makes the transformation.
    
    input: full_dict: full gene reviews dictionary
    
    output: full_dict: full gene reviews dictionary after name transformation
    '''
    suffix_list = ['Jr', 'II', 'III', 'IV']
    for i in full_dict:
        if ',' in full_dict[i]['authors']:
            full_dict[i]['authors'] = full_dict[i]['authors'].replace('and ', '').strip('.').split(', ')
        else:
            full_dict[i]['authors'] = full_dict[i]['authors'].replace(' and', ',').strip('.').split(', ')
    for i in full_dict:
        for j in range(len(full_dict[i]['authors'])):
            full_dict[i]['authors'][j] = full_dict[i]['authors'][j].split(' ')
 
    #two cases: lowercase and suffix
    for i in full_dict:
        count = 0
        for j in full_dict[i]['authors']:
            name = ''  #Final name will be stored
            last_name = ''
            suffix = ''
            k = 0
            if len(j) < 7:
                while k < len(j):
                    if any(x in j for x in suffix_list):
                        if (not j[k].islower()) and (k != len(j) - 1) and (k != len(j) - 2):
                            j[k] = remove_lower(j[k])
                            if len(j[k]) >= 2 and '-' not in j[k]:
                                j[k] = '. '.join(j[k]).strip()
                        elif k == len(j) -2:
                            last_name = j[k]
                        else:
                            suffix = j[k]
                    else:
                        if (not j[k].islower()) and (k != len(j) - 1):
                            j[k] = remove_lower(j[k])
                            if len(j[k]) >= 2 and '-' not in j[k]:
                                j[k] = '. '.join(j[k]).strip()
                        elif k == len(j) - 1:
                            last_name = j[k]                        
                    k += 1
                    
                for k in range(len(j)):
                    if any(x in j for x in suffix_list):
                        if (not j[k].islower()) and (k != len(j) - 1) and (k != len(j) - 2):
                            name += (j[k] + '. ')
                        elif j[k].islower():
                            last_name = ''
                            for l in range(k, len(j) - 1):
                                last_name += (j[l] + ' ')
                            break
                    else:
                        if (not j[k].islower()) and (k != len(j) - 1):
                            name += (j[k] + '. ')
                        elif j[k].islower():
                            last_name = ''
                            for l in range(k, len(j)):
                                last_name += (j[l] + ' ')
                            break                        
                    
                if suffix != '':
                    name = last_name.strip() + ', ' + name.strip() + ', ' + suffix + '.'
                else:
                    name = last_name.strip() + ', ' + name.strip()
            #print(i)
            #print(full_dict[i]['omimID_list'])
            full_dict[i]['authors'][count] = name
            #print(name)
            #print('\n')
            count += 1
    return full_dict

def num_with_no_omimID(full_dict):
    '''
    This function returns the number of gene review articles with no related omimID.
    Input: gr_dict: gene reviews full dictionary
    OUtput: count: the number of articles that do not have related omimID
    '''
    count = 0
    list_no_related = []
    for i in full_dict:
        if full_dict[i]['omimID_list'] == []:
            count += 1
            list_no_related.append(i)
            print(i)  
    return list_no_related

def num_no_pub(gr_dict, omim_dict):
    '''
    This fucntion is to count the number of authors in genreviews who do not have publications
    on OMIM. And also returns the people have publications in a list called training_dict which
    is the initial dict for training
    
    '''
    count = 0
    count1 = 0
    count2 = [] #publication number count
    name = []
    disease_name = []
    training_dict = {}
    for disease in gr_dict:
        for omimID in gr_dict[disease]['omimID_list']:
            for author in gr_dict[disease]['authors']:
                count1 += 1
                count3 = 0
                flag = 0
                try:
                    for pub in omim_dict[omimID]['pubList']:
                        if author in omim_dict[omimID]['pubList'][pub]['authors']:
                            flag = 1
                            count3 += 1
                            if omimID not in training_dict:
                                training_dict[omimID] = {'authors': {author: 1}, 'disease': disease}
                            else:
                                if author not in training_dict[omimID]['authors'].keys():
                                    training_dict[omimID]['authors'][author] = 1
                                else:
                                    training_dict[omimID]['authors'][author] += 1                            
                    if flag == 0:
                        count += 1
                except KeyError:
                    pass
                if count3 != 0:
                    count2.append(count3)
                    name.append(author)
                    disease_name.append(omimID)
    return count, count1, count2, name, disease_name, training_dict

def get_omim_num(full_dict):
    count = 0
    for i in full_dict:
        count += len(full_dict[i]['omimID_list']) 
    return count

def get_initial_gene_review_training_dict(gr_dict, omim_dict):
    training_dict = {}
    for disease in gr_dict:
        for omimID in gr_dict[disease]['omimID_list']:
            for author in gr_dict[disease]['authors']:
                try:
                    for pub in omim_dict[omimID]['pubList']:
                        if author in omim_dict[omimID]['pubList'][pub]['authors']:
                            if omimID not in training_dict:
                                training_dict[omimID] = {'authors': {author: 1}, 'disease': disease}
                            else:
                                if author not in training_dict[omimID]['authors'].keys():
                                    training_dict[omimID]['authors'][author] = 1
                                else:
                                    training_dict[omimID]['authors'][author] += 1  
                except KeyError:
                    pass                    
    return training_dict
            
    
if __name__ == '__main__':
    
    #Step 1: open the file generated from gen_reviews_scraping.py
    with open('GR_disease_author_OMIMid_dict.json', 'r') as f: #for gene review
        gr_dict = json.load(f)
    
    #Step 2: open the file generated from omim_scraping.py    
    with open('omim_dict.json', 'r') as f:
        omim_dict = json.load(f)    
    
    #Step 3: transform author names in GeneReviews into OMIM name format
    gr_dict = transform_names_to_omim_format(gr_dict)
    
    #Step 4: get initial positive labeled data with publication numbers
    #this dict is for the authors in gene reveiws who has publications on the rare diseases
    gene_reviews_training_dict = get_initial_gene_review_training_dict(gr_dict, omim_dict) 
    
    with open('gene_reviews_training_dict.json', 'w') as f:
        json.dump(gene_reviews_training_dict, f)