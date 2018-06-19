import math



def tf_idf(corpus):
    lst = []
    voc = {}
    index = 0
    for text in corpus:
        tokens = text.strip().split("|")
        for t in tokens:
            if t in voc:
                continue
            else:
                voc[t] = index
                index += 1
        lst.append(tokens)

    #print lst
    vector = []
    for text in lst:
        vec = [0] * len(voc)
        for t in text:
            #calculate term frequency
            index = voc[t]
            vec[index] += 1

        m = float(max(vec))
        norm_vec = [ x / m for x in vec]
        vector.append(norm_vec)
    #print vector
    #idf
    idf = [0] * len(voc)
    for v in vector:
        for (i,c) in enumerate(v):
            if c!=0:
                idf[i] += c/c
            else:
                continue
    N = float(len(vector))
    idf = [math.log(1 + N/x, 2) for x in idf]
    #print idf

    #tf-idf
    lst_tfidf = []
    
    for v in vector:
        max_tf = float(max(v))
        tfidf = [round(idf[i] * float(c) ,2) for (i,c) in enumerate(v)]
        lst_tfidf.append(tfidf)
    #print lst_tfidf
    return lst_tfidf

def tfidf_similarity(vec1, vec2):
    product = 0
    euclidean1 = math.sqrt(sum([x**2 for x in vec1]))
    euclidean2 = math.sqrt(sum([x**2 for x in vec2]))
    for (i ,t) in enumerate(vec1):
        product += float(t) * float(vec2[i])
    similarity = float(product)/(euclidean1*euclidean2)
    return similarity

if  __name__=="__main__":
    corpus = ['a|b|c|f',
              'c|d|e',
              'a|b|b|c',
              'f|a|b|e|c']
    vecs = tf_idf(corpus)



