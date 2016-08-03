import json
import string
import re

def remove_lower(string):
    new = ''
    remove_low = lambda text: re.sub('[a-z]', '', text)
    new = remove_low(string)
    return new

def transform_names_to_omim_format(full_dict):
    
    
    suffix_list = ['Jr', 'II', 'III', 'IV']
    
    for i in full_dict:
        if ',' in full_dict[i]['authors']:
            full_dict[i]['authors'] = full_dict[i]['authors'] .replace('and ', '').strip('.').split(', ')
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
            
                         
    return 

def num_with_no_omimID(gr_dict):
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
    on OMIM
    
    '''
    count = 0
    count1 = 0
    

    
    for disease in gr_dict:
        for omimID in gr_dict[disease]['omimID_list']:
            for author in gr_dict[disease]['authors']:
                count1 += 1
                flag = 0
                try:
                    for pub in omim_dict[omimID]['pubList']:
                        if author in omim_dict[omimID]['pubList'][pub]['authors']:
                            flag = 1
                    if flag == 0:
                        print(author)
                        print(omimID)
                        print(disease)
                        count += 1
                except KeyError:
                    pass
    
                    
    return count, count1

def get_omim_num(full_dict):
    count = 0
    
    for i in full_dict:
        count += len(full_dict[i]['omimID_list']) 
        
    return count
            
    
    
    




if __name__ == '__main__':

    with open('disease_author_omimID_dict.json', 'r') as f:
        full_dict = json.load(f)
        
    with open('omim_dict_final.json', 'r') as f:
        omim_dict = json.load(f)    
        
    transform_names_to_omim_format(full_dict)
    
    
    num, count = num_no_pub(full_dict, omim_dict)
    
    #for i in full_dict:
        #for j in full_dict[i]['authors']:
            #for k in j:
                #if k.islower() and len(j) < 7:
                    #print(k, len(j), j, i)
            
    
    
            
            
    #for i in full_dict:
        #for j in full_dict[i]['authors']:
            #if any(x in j for x in suffix_list):
                #print(j)
            
    
    #for i in full_dict:
