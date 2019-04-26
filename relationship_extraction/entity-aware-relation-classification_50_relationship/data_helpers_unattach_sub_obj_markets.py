import numpy as np
import pandas as pd
import nltk
import re
from tqdm import tqdm
import os
import json
from tqdm import tqdm
import subprocess
import utils
from configure import FLAGS
from bert_serving.client import BertClient
import multiprocessing


def clean_str(text):
    text = text.lower()
    # Clean the text
    # text = re.sub(r"_", " ", text)
    # text = re.sub(r"what's", "what is ", text)
    # text = re.sub(r"that's", "that is ", text)
    # text = re.sub(r"there's", "there is ", text)
    # text = re.sub(r"it's", "it is ", text)
    # text = re.sub(r"\'s", " ", text)
    # text = re.sub(r"\'ve", " have ", text)
    # text = re.sub(r"can't", "can not ", text)
    # text = re.sub(r"n't", " not ", text)
    # text = re.sub(r"i'm", "i am ", text)
    # text = re.sub(r"\'re", " are ", text)
    # text = re.sub(r"\'d", " would ", text)
    # text = re.sub(r"\'ll", " will ", text)
    # text = re.sub(r",", " ", text)
    # text = re.sub(r"\.", " ", text)
    # text = re.sub(r"!", " ! ", text)
    # text = re.sub(r"\/", " ", text)
    # text = re.sub(r"\^", " ^ ", text)
    # text = re.sub(r"\+", " + ", text)
    # text = re.sub(r"\-", " - ", text)
    # text = re.sub(r"\=", " = ", text)
    # text = re.sub(r"'", " ", text)
    # text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    # text = re.sub(r":", " : ", text)
    # text = re.sub(r" e g ", " eg ", text)
    # text = re.sub(r" b g ", " bg ", text)
    # text = re.sub(r" u s ", " american ", text)
    # text = re.sub(r"\0s", "0", text)
    # text = re.sub(r" 9 11 ", "911", text)
    # text = re.sub(r"e - mail", "email", text)
    # text = re.sub(r"j k", "jk", text)
    # text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def load_data_and_labels(path, features_saved_path, type):
    if not os.path.exists(features_saved_path + type + "_features.csv"):
        data = []
        relation_all_true = []
        lines = []
        with open(path, 'r') as f:
            f.readline()
            for line in tqdm(f.readlines()):
                lines.append(line.strip())
        max_sentence_length = 0
        for idx in tqdm(range(0, len(lines), 3)):
            id = lines[idx].split("\t")[0]
            relation = lines[idx + 1]
            relation_all_true.append(relation)

            sentence = lines[idx].split("\t")[2][1:-1]
            #sentence = clean_str(sentence)
            subject = 0
            object = 0
            try:
                tokens = nltk.word_tokenize(sentence)
                if max_sentence_length < len(tokens):
                    max_sentence_length = len(tokens)
                # subject = tokens.index("subject2") - 1
                # object = tokens.index("object2") - 1
                subject = int(lines[idx].split("\t")[5])
                object = int(lines[idx].split("\t")[8])
                sentence = " ".join(tokens)
            except:
                print(sentence)

            data.append([id, sentence, subject, object, relation])
        if not os.path.exists("./resource/dev_target.txt") and type == "dev":
            with open("./resource/dev_target.txt", "w") as dev_relation_truely_output:
                for i, re in enumerate(relation_all_true):
                    dev_relation_truely_output.write(str(i) + " " + re + "\n")
                dev_relation_truely_output.close()

        print(path)
        print("max sentence length = {}\n".format(max_sentence_length))
        print("please wait for a moment.")
        df = pd.DataFrame(data=data, columns=["id", "sentence", "subject", "object", "relation"])

        pos1, pos2 = get_relative_position(df, FLAGS.max_sentence_length)
        df["pos1"] = pos1
        df["pos2"] = pos2
        df['label'] = [utils.class2label[r] for r in df['relation']]
        df.to_csv(features_saved_path + type + "_features.csv", sep="\t", index=False)
    else:
        df = pd.read_csv(features_saved_path + type + "_features.csv", sep="\t")
        print(df.columns)
        pos1 = df["pos1"].tolist()
        pos2 = df["pos2"].tolist()
        relation_all_true = []
        if type == "dev":
            with open("./resource/dev_target.txt", "r") as dev_relation_truely_in:
                for li in dev_relation_truely_in:
                    relation_all_true.append(li.split(" ")[1].replace("\n", ''))
                dev_relation_truely_in.close()
        else:
            relation_all_true = []

    # Text Data
    x_text = df['sentence'].tolist()
    subject = df['subject'].tolist()
    object = df['object'].tolist()

    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 49 => [0 0 0 0 0 ... 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    return x_text, labels, subject, object, pos1, pos2, relation_all_true


def get_relative_position(df, max_sentence_length):
    # Position data
    pos1 = []
    pos2 = []
    for df_idx in tqdm(range(len(df))):
        sentence = df.iloc[df_idx]['sentence']
        tokens = nltk.word_tokenize(sentence)
        subject = df.iloc[df_idx]['subject']
        object = df.iloc[df_idx]['object']

        p1 = ""
        p2 = ""
        for word_idx in range(len(tokens)):
            p1 += str((max_sentence_length - 1) + word_idx - subject) + " "
            p2 += str((max_sentence_length - 1) + word_idx - object) + " "
        pos1.append(p1)
        pos2.append(p2)

    return pos1, pos2


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_bert_embed(vocabulary_path, bert_vec_output_path):

    def generate_bert_vec(words_list_, num_):
        with open(bert_vec_output_path + str(num_), "w") as output:
            for w in tqdm(words_list_):
                vector = bc.encode([w])
                str_ = ''
                for i in vector.tolist()[0]:
                    str_ += str(i) + ' '
                str_line = str_.strip(' ')
                output.write(w + " " + str_line + '\n')
            output.close()

    bc = BertClient()
    words_list = []
    print("generate bert word vector...")
    with open(vocabulary_path, "r") as vocab:
        vocab_dict = json.load(vocab)
        for key in tqdm(vocab_dict.keys()):
            words_list.append(key)
        vocab.close()
        split_num = int(len(words_list)/5000)
        for num in range(28, split_num):
            words_temp = words_list[num*5000:(num+1)*5000]
            generate_bert_vec(words_temp, num)
            #p = multiprocessing.Process(target=generate_bert_vec, args=(words_temp, num))
            print("start job...")
            #p.start()
        words_leftover = words_list[split_num*5000:]
        generate_bert_vec(words_leftover, split_num)


def cat_bert_vec():
    home_name = "bert_vector.d768-"
    file_name = ""
    for i in range(81):
        file_name += home_name + str(i) + " "
    file_name = file_name.strip()
    subprocess.call("cat " + file_name +" > lic2019_bert_vector.768d.txt", shell=True)
    print("finished! and are be saved at : ./lic2019_bert_vector.768d.txt")


if __name__ == "__main__":
    # trainFile = './entity_aware_dataset_lic2019_information_extraction/entity_aware_train'
    devFile = './entity_aware_dataset_lic2019_information_extraction/' \
              'classification_3.0_50_relationship_update_20190418/entity_aware_dev'
    #
    # load_data_and_labels(trainFile)
    load_data_and_labels(devFile, "./entity_aware_dataset_lic2019_information_extraction/"
                         "classification_3.0_50_relationship_update_20190418/", type="dev")
    # with open("./vocab_word2id.dict", "r", encoding="utf-8") as f:
    #     dict = json.load(f)
    #     for k, v in dict.items():
    #         print(str(k) + " " + str(v)) ##何庆成 89946

    #get_bert_embed(vocabulary_path="./vocab_word2id.dict", bert_vec_output_path="./bert_vector.d768-")
    #cat_bert_vec()
    # df = pd.read_csv("./dev_features.csv", sep="\t")
    # print(df["pos1"])
