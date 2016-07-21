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
            
            j = name
            print(j)
                    
                    
                
            
        
            
    return 
            
    
    
    




if __name__ == '__main__':

    with open('disease_author_omimID_dict.json', 'r') as f:
        full_dict = json.load(f)
        
    transform_names_to_omim_format(full_dict)
    
    for i in full_dict:
        for j in full_dict[i]['authors']:
            for k in j:
                if k.islower() and len(j) < 7:
                    print(k, len(j), j, i)
            
    
    
            
            
    #for i in full_dict:
        #for j in full_dict[i]['authors']:
            #if any(x in j for x in suffix_list):
                #print(j)
            
    
    #for i in full_dict:
