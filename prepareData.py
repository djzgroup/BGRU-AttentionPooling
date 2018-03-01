import numpy as np
import random as rd


def replace_symbol(sentence):
    Symbol = ['"',"#","$","%","&","(",")","*","+","-",".","/",":",";","<","=",">","?","@","[","\\","]","^","_","`","{","|","}","\t","\n",","]
    sentence = sentence.replace('-rrb-', ")").replace("-lrb-", "(").lower()
    for i in range(len(Symbol)):
        sentence = sentence.replace(Symbol[i],'')
    return sentence

def mapping_vector(data_GLOVE_file,word_index,embeddings_size):
    embeddings_index={}
    with open(data_GLOVE_file,encoding="utf-8")as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddings_data = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embeddings_data
    print("Found%s word vector"%len(embeddings_index))
    word_size = len(word_index.keys())
    embeddings_matrix = np.zeros((word_size,embeddings_size))
    for word in word_index.keys():
        if word in embeddings_index.keys():
            embeddings_matrix[word_index[word]]=embeddings_index[word]
        else:
            embeddings_matrix[word_index[word]]=np.random.uniform(-0.1, 0.1, embeddings_size)
    embeddings_matrix = np.array(embeddings_matrix)
    return embeddings_matrix

def Statistical_length(text):
    #Count the length of the longest sentence in the data set

    max_length=0
    for i in range(len(text)):
            if max_length<len(text[i]):
                max_length = len(text[i])
    return max_length

def padding_sentence(data,max_length=65):
    #pad any sentence smaller than the longest length in the data set

    for i in range(len(data)):
        if len(data[i])>max_length:
            data[i]=data[i][:max_length]
        elif len(data[i])<max_length:
            pad_num = max_length-len(data[i])
            for m in range(pad_num):
                data[i].append(0)
        else:
            pass
    return data
def concatenate(data,label):
    sent =[]
    if len(data)==len(label):
        for i in range(len(data)):
            sent.append([data[i],label[i]])
    else:
        raise ValueError("The data set size %d is not the same as the label size %d."% len(data) %len(label))
    return np.array(sent)

def Tagging(tag,classes):
    #Marked the label of the sentence, currently only wrote two categories,
    # if there are other species classification, according to the following code to correct it

    if classes==2:
        if tag=="0":
            return [1,0]
        else:
            return [0,1]
    elif classes==5:
        if tag=="0":
            return [1,0,0,0,0]
        elif tag=="1":
            return [0,1,0,0,0]
        elif tag=="2":
            return [0,0,1,0,0]
        elif tag=="3":
            return [0,0,0,1,0]
        elif tag=="4":
            return [0,0,0,0,1]
    else:
        raise ValueError("Unknown parameters %d." % classes)



def processing_data(data_file,classes):
    data=[]
    word_index={}
    count = 1
    label =[]
    with open(data_file,encoding="utf-8")as f:
        sentencs = f.readlines()
        for sent in sentencs:
            sent_data = []
            sent = replace_symbol(sent)
            sent = sent.split(" ")
            for word in sent[1:]:
                if word=='':
                    continue
                if word not in word_index.keys():
                    word_index[word]=count
                    count = count+1
                sent_data.append(word_index[word])
            label.append(Tagging(sent[0],classes=classes))
            data.append(sent_data)
    word_index["UNK"] = 0
    return data,word_index,label

def segment_dataset(training_ratio,data):

    data_number = len(data)
    data = np.array(data)
    np.random.shuffle(data)

    n =(1-training_ratio)*data_number
    test_index=[]
    while (n>0):
        index = rd.randint(0,data_number-1)
        if index not in test_index:
            test_index.append(index)
            n =n-1
    data_index =np.arange(data_number)
    train_index = [m for m in data_index if m not in test_index]
    train =[data[i] for i in train_index]
    test =[data[i] for i in test_index]

    # divide the data into index of sentence and label
    train_data =[x[0] for x in train]
    train_label = [x[1] for x in train]
    test_data = [x[0] for x in test]
    test_label = [x[1] for x in test]

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    return train_data,train_label,test_data,test_label



if __name__ == "__main__":
    data_file =r""
    data_GLOVE_file=r""
    data,word_index,label = processing_data(data_file, 2)
    max_length = Statistical_length(data)
    print("the max length is %d" % max_length)
    data = padding_sentence(data,66)
    data = concatenate(data,label)
    print(data.shape)
    embeddings_matrix = mapping_vector(data_GLOVE_file, word_index, 300)
    embeddings_matrix = np.array(embeddings_matrix)
    train_data, train_label, test_data, test_label =segment_dataset(0.9, data)
    np.save("data/train_data.npy",train_data)
    np.save("data/train_label.npy",train_label)
    np.save("data/test_data.npy",test_data)
    np.save("data/test_label.npy", test_label)
    np.save("data/embeddings_matrix.npy", embeddings_matrix)
