#!/usr/bin/env python

import sys
import pickle
from tf import tf_idf,tfidf_similarity


train_file = open(sys.argv[1],"r")
test_file = open(sys.argv[2], "r")
#to_write =open(sys.argv[3],"w+")



feature_map = {}
feature_index = 0


HASH_SIZE = 1000000

def process_id_feature(prefix, id):
     global feature_map
     global feature_index

     str = prefix + "_" + id
     if str in feature_map:
          return feature_map[str]
     else:
          feature_index = feature_index + 1
          feature_map[str] = feature_index
     return feature_index


def hash_feature(prefix, id):
    str = prefix + "_" + id
    return hash(str)% HASH_SIZE


def extract_feature1(seg):
    lst = []
    
    lst.append(process_id_feature("url", seg[1]))
    lst.append(process_id_feature("ad", seg[2]))

    lst.append(process_id_feature("adver", seg[3]))

    lst.append(process_id_feature("depth", seg[4]))
    lst.append(process_id_feature("pos", seg[10]))

    lst.append(process_id_feature("query", seg[5]))

    lst.append(process_id_feature("keyword", seg[6]))
    lst.append(process_id_feature("title", seg[7]))
    lst.append(process_id_feature("user", seg[9]))
    lst.append(process_id_feature("description", seg[8]))
    return lst



def extract_feature2(seg):
    depth = float(seg[4])
    pos = float(seg[10])
    relative_pos = int(pos*10/depth)
    return process_id_feature("pos_ratio", str(relative_pos))

def extract_combination_feature(seg):
    lst = []
    if(len(seg) >= 16):
         str = seg[2] + "_" + seg[15]
         lst.append(process_id_feature("user_gender", str))
         
         str1 = seg[15] + "_" + seg[16]
         lst.append(process_id_feature("gender_age", str1))
         
         str4 = seg[9] + "_" +seg[16]
         lst.append(process_id_feature("user_age", str4))
         
    str2 = seg[2] + "_" + seg[5]
    lst.append(process_id_feature("query_ad", str2))

    str3 = seg[2] + "_" + seg[9]
    lst.append(process_id_feature("ad_user",str3))
    
    #str5 = seg[] + "_" + seg[]
    return lst

def extract_numerical_feature(seg):
    lst = []
    query_len = len(seg[11].strip().split("|"))
    keyword_len = len(seg[12].strip().split("|"))	
    title_len = len(seg[13].strip().split("|"))
    description_len = len(seg[14].strip().split("|"))
    lst.append(str(process_id_feature("query_len", " ")) + ":" + str(query_len))
    lst.append(str(process_id_feature("keyword_len", " ")) + ":" + str(keyword_len))
    lst.append(str(process_id_feature("title_len", " ")) + ":" + str(title_len))
    lst.append(str(process_id_feature("desc_len", " ")) + ":" + str(description_len))
    #extract tfidf feature of all tokens
    
    corpus = seg[11:15]
    tfidf = tf_idf(corpus)

    
    query_similar_keyword = tfidf_similarity(tfidf[0],tfidf[1])
    query_similar_tile = tfidf_similarity(tfidf[0],tfidf[2])
    query_similar_description = tfidf_similarity(tfidf[0],tfidf[3])
    keyword_similar_title = tfidf_similarity(tfidf[1],tfidf[2])
    keyword_similar_description = tfidf_similarity(tfidf[1],tfidf[3])
    title_similar_description = tfidf_similarity(tfidf[2],tfidf[3])

    lst.append(str(process_id_feature("query_similar_keyword", " ")) + ":" + str(query_similar_keyword) )
    lst.append(str(process_id_feature("query_similar_tile", " ")) + ":" + str(query_similar_tile) )
    lst.append(str(process_id_feature("query_similar_description", " ")) + ":" + str(query_similar_description ) )
    lst.append(str(process_id_feature("keyword_similar_title", " ")) + ":" + str(keyword_similar_title ) )
    lst.append(str(process_id_feature("keyword_similar_description", " ")) + ":" + str(keyword_similar_description ) )
    lst.append(str(process_id_feature("title_similar_description", " ")) + ":" + str(title_similar_description ) )
    
    lst.append(str(process_id_feature("sum_idf_query", " ")) + ":" + str(sum(tfidf[0]) ) )
    lst.append(str(process_id_feature("sum_idf_keyword", " ")) + ":" + str( sum(tfidf[1]) ))
    lst.append(str(process_id_feature("sum_idf_title", " ")) + ":" + str(sum(tfidf[2]) ) )
    lst.append(str(process_id_feature("sum_idf_description", " ")) + ":" + str(sum(tfidf[3]) ) )

    
    
    depth = float(seg[4])
    postion = float(seg[10])
    relative_pos = float((depth-postion)*10.0/depth)
                  
    lst.append(str(process_id_feature("depth_num", " ")) + ":" + str(depth))
    lst.append(str(process_id_feature("postion_num", " ")) + ":" + str(postion))
    lst.append(str(process_id_feature("relative_pos_num", " ")) + ":" + str(relative_pos))
    
    """
    raw_query = int(seg[5])
    raw_user = int(seg[9])

    lst.append(str(process_id_feature("raw_query"," ")) + ":" + str(raw_query))
    lst.append(str(process_id_feature("raw_user"," ")) + ":" + str(raw_user))
    """
    

    return lst



def cate_to_str(label, lst):
    line = label
    for i in lst:
         line = line + "\t" + str(i) + ":1"
    return line

def numer_to_str(numer_lst):
    return "\t".join(numer_lst)

name = ['train','test']
i = 0
for file in [train_file,test_file]:

    to_write = open(str(name[i]) + "_feature", "w+")
    for line in file:
        seg = line.strip().split("\t")
        lst_cate = extract_feature1(seg)
        lst_cate.append(extract_feature2(seg))
        lst_cate.extend(extract_combination_feature(seg))
        lst_numer = extract_numerical_feature(seg)
    
        to_write.write(cate_to_str(seg[0], lst_cate) + "\t" + numer_to_str(lst_numer) + "\n")
    to_write.close()
    i += 1


