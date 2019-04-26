import json
import re
import os
import re
from tqdm import tqdm
import pandas as pd
from gensim.models import Word2Vec
import jieba
import pkuseg
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


class DataReader(object):
    """
    class for data reader and data analyse
    """
    def __init__(self,
                 train_path,
                 dev_path,
                 test_path):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self._p_map_eng_dict = {}

    def get_all_relationship(self,):
        all_rel = []
        train_rel = []
        with open(self.train_path, 'r', encoding='utf-8') as f:
            logger.info("processing training set")
            for line in tqdm(f.readlines()):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                try:
                    dict = json.loads(line.strip())
                    if "spo_list" not in dict:
                        continue

                    for dict_ in dict["spo_list"]:
                        train_rel.append(dict_["predicate"])
                        all_rel.append(dict_["predicate"])
                except:
                    print("exist error in the training set, please check!")

        dev_rel = []
        with open(self.dev_path, 'r', encoding='utf-8') as dev_f:
            logger.info("processing dev set")
            for line in tqdm(dev_f.readlines()):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                try:
                    dict = json.loads(line.strip())
                    if "spo_list" not in dict:
                        continue

                    for dict_ in dict["spo_list"]:
                        dev_rel.append(dict_["predicate"])
                        if dict_["predicate"] not in all_rel:
                            all_rel.append(dict_["predicate"])
                except:
                    print("exist error in the dev set, please check!")
        return set(all_rel), train_rel, dev_rel

    def get_relationship_zh2en_dict(self, input_path):
        zh2en_dict = {}
        with open(input_path, "r") as f:
            for li in f.readlines():
                li_ = li.replace("\n", '').split("\t")
                zh2en_dict[li_[0]] = li_[1]
        return zh2en_dict

    def generate_relationship_quantity_distribution_diagram(self, zh2en_dict_path):
        zh2en_dict = self.get_relationship_zh2en_dict(zh2en_dict_path)
        category_rel, train_rel, dev_rel = self.get_all_relationship()
        train_rel_zh2en = []
        for item in train_rel:
            train_rel_zh2en.append(zh2en_dict.get(item))
        train_rel_ = pd.Series(train_rel_zh2en)
        cnt_srs_train = train_rel_.value_counts()
        plt.figure(figsize=(50, 8))
        plt.tick_params(labelsize=7)
        plt.bar(cnt_srs_train.index, cnt_srs_train.values, alpha=0.9, width=0.5, facecolor='lightskyblue',
                edgecolor='white', label='train', lw=1)

        for a, b in zip(cnt_srs_train.index, cnt_srs_train.values):  # Display corresponding number
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=6)
        plt.legend(loc="upper right")
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Relationship Labels', fontsize=12)
        plt.title("The relationship label statistical graph of the training set")
        plt.xticks(rotation='30')
        plt.show()

        dev_rel_zh2en = []
        for item in dev_rel:
            dev_rel_zh2en.append(zh2en_dict.get(item))
        dev_rel_ = pd.Series(dev_rel_zh2en)
        cnt_srs_dev = dev_rel_.value_counts()
        plt.figure(figsize=(50, 8))
        plt.tick_params(labelsize=7)
        plt.bar(cnt_srs_dev.index, cnt_srs_dev.values, alpha=0.9, width=0.5, facecolor='limegreen',
                edgecolor='white', label='dev', lw=1)
        for a, b in zip(cnt_srs_dev.index, cnt_srs_dev.values):  # Display corresponding number
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=6)
        plt.legend(loc="upper right")
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Relationship Labels', fontsize=12)
        plt.title("The relationship label statistical graph of the dev set")
        plt.xticks(rotation='30')
        plt.show()

    def _valid_sub_obj_intersection(self,):
        sub_all_train = []
        obj_all_train = []
        train_path = "./data/train_data.json"
        with open(train_path, 'r') as f:
            for line in f:
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                line = line.strip()
                dic = json.loads(line)
                spo_list = dic["spo_list"]
                sub_all_train.extend([str(spo['subject']).replace(" ", '').replace("《", '').replace("》", '')
                                for spo in spo_list])
                obj_all_train.extend([str(spo['object']).replace(" ", '').replace("《", '').replace("》", '')
                                for spo in spo_list])
        print("The number of subjects in the training set is: " + str(len(set(sub_all_train))) + "," +
              "The number of objects in the training set is: " + str(len(set(obj_all_train))))
        intersection_set_train = set(sub_all_train).intersection(set(obj_all_train))
        print("The number of overlapping subjects and objects in the training set is: " +
              str(len(intersection_set_train)))  ##output: 8698

        sub_all_dev = []
        obj_all_dev = []
        dev_path = "./data/dev_data.json"
        with open(dev_path, 'r') as f:
            for line in f:
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                line = line.strip()
                dic = json.loads(line)
                spo_list = dic["spo_list"]
                sub_all_dev.extend([spo["subject"] for spo in spo_list])
                obj_all_dev.extend([spo["object"] for spo in spo_list])
        print("The number of subjects in the dev set is: " + str(len(sub_all_dev))
              + "," +
              "The number of objects in the dev set is: " + str(len(obj_all_dev)))
        intersection_set_dev = set(sub_all_dev).intersection(set(obj_all_dev))
        print("The number of overlapping subjects and objects in the dev set is: " +
              str(len(intersection_set_dev)))  # output: 1536

        return intersection_set_train, intersection_set_dev

    def analyse_text_length_in_raw_data(self,):

        text_length = []
        logger.info("processing training set...")
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                dic = line.strip()
                dic = json.loads(dic)
                text_length.append(len(dic["text"]))
        logger.info("processing dev set...")
        with open(self.dev_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                dic = line.strip()
                dic = json.loads(dic)
                text_length.append(len(dic["text"]))

        text_length_ = pd.Series(text_length)
        cnt_srs = text_length_.value_counts()
        logger.info("The maximum length of the text in the training set and dev set is: " + str(text_length_.max()))  ##300
        logger.info("The average text length in the training set and dev set is: " + str(text_length_.mean()))  ##54.66833378691328
        logger.info("The std text length in the training set and dev set is: " + str(text_length_.std()))  ## 32.07152057874568

        plt.figure(figsize=(20, 8))
        plt.bar(cnt_srs.index, cnt_srs.values, alpha=0.9, width=0.8, facecolor='limegreen',
                edgecolor='white', label='train & dev', lw=1)
        # for a, b in zip(cnt_srs.index, cnt_srs.values):  # Display corresponding number
        #     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
        # sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('The length of the text', fontsize=12)
        plt.title("The length of the text in training set and dev set")
        plt.xticks(rotation='horizontal')
        plt.legend(loc="upper right")
        plt.show()

    def after_split_text_analyse_text_length_in_raw_data(self, train_output_path):
        logger.info("processing training set...")
        with open(self.train_path, 'r', encoding='utf-8') as f:
            train_output = open(train_output_path, 'w')
            train_output.write("row_id" + '\t' + "text" + '\t' + "subject" + '\t'
                               + "object" + '\t' + "relationship" + '\n')
            row_id = 0
            for line in tqdm(f.readlines()):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                dic = line.strip()
                dic = json.loads(dic)
                text = str(dic["text"])
                if "spo_list" not in dic:
                    continue
                for spo in dic["spo_list"]:
                    p = spo["predicate"]
                    sub = spo["subject"]
                    obj = spo["object"]
                    train_output.write(str(row_id) + '\t' + text + '\t' + sub + '\t' + obj + '\t' + p + '\n')
                row_id += 1
            train_output.close()

    def analyse_sub_obj_in_raw_data(self,):
        logger.info("Processing training set")
        count_train = 0
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                dic = line.strip()
                dic = json.loads(dic)
                if "spo_list" not in dic:
                    continue
                sub_temp = set()
                obj_temp = set()
                for spo in dic["spo_list"]:
                    sub = spo["subject"]
                    sub_temp.add(sub)
                    obj = spo["object"]
                    obj_temp.add(obj)
                if len(sub_temp) > 1 and len(obj_temp) > 1:
                    count_train += 1

        logger.info("Processing dev set")
        count_dev = 0
        with open(self.dev_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                dic = line.strip()
                dic = json.loads(dic)
                if "spo_list" not in dic:
                    continue
                sub_temp = set()
                obj_temp = set()
                for spo in dic["spo_list"]:
                    sub = spo["subject"]
                    sub_temp.add(sub)
                    obj = spo["object"]
                    obj_temp.add(obj)
                if len(sub_temp) > 1 and len(obj_temp) > 1:
                    count_dev += 1

        print("The total number of examples with multiple subjects and multiple objects in a text in the training set "
              "is: \n" + str(count_train))  # outpot:14325
        print("The total number of examples with multiple subjects and multiple objects in a text in the dev set is: \n"
              + str(count_dev))  # output:1768

    def statistical_sub_obi_ner_labels_category_nummbers(self, train_ner_path, dev_ner_path):

        train_labels_category = set()
        train_labels_all = []
        with open(train_ner_path, 'r') as train_f:
            for li in tqdm(train_f.readlines()):
                li_ = li.split(" ")
                if li_[1] == '':
                    continue
                train_labels_category.add(str(li_[1]).replace("\n", ''))
                train_labels_all.append(str(li_[1]).replace("\n", ''))

        dev_labels_category = set()
        dev_labels_all = []
        with open(dev_ner_path, 'r') as dev_f:
            for li in tqdm(dev_f.readlines()):
                li_ = li.split(" ")
                if li_[1] == '':
                    continue
                dev_labels_category.add(str(li_[1]).replace("\n", ''))
                dev_labels_all.append(str(li_[1]).replace("\n", ''))
        print("In the training set, ner labels category numbers: " + str(len(train_labels_category)) + "\n"
              + str(list(train_labels_category)))
        # output:
        # In the training set, ner labels category numbers: 7
        # ['B-OBJ', 'I-SUB', 'O', 'B-SUB', 'I-OBJ', 'I-SUB_OBJ', 'B-SUB_OBJ']
        print("\nIn the dev set, ner labels category numbers: " + str(len(dev_labels_category)) + "\n"
              + str(list(dev_labels_category)))
        # output:
        # In the dev set, ner labels category numbers: 7
        # ['B-OBJ', 'I-SUB', 'O', 'B-SUB', 'I-OBJ', 'I-SUB_OBJ', 'B-SUB_OBJ']

        # Draw the ner label statistical graph of the training set
        train_labels_all_ = pd.Series(train_labels_all)
        cnt_srs_train = train_labels_all_.value_counts()
        plt.figure(figsize=(20, 8))
        plt.bar(cnt_srs_train.index, cnt_srs_train.values, alpha=0.9, width=0.35, facecolor='lightskyblue',
                edgecolor='white', label='train', lw=1)
        for a, b in zip(cnt_srs_train.index, cnt_srs_train.values):  # Display corresponding number
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
        plt.legend(loc="upper right")

        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Ner Label', fontsize=12)
        plt.title("The ner label statistical graph of the training set")
        # plt.xticks(rotation='vertical')
        plt.show()

        # Draw the ner label statistical graph of the training set
        dev_labels_all_ = pd.Series(dev_labels_all)
        cnt_srs_dev = dev_labels_all_.value_counts()
        plt.figure(figsize=(20, 8))
        plt.bar(cnt_srs_dev.index, cnt_srs_dev.values, alpha=0.9, width=0.35, facecolor='limegreen',
                edgecolor='white', label='dev', lw=1)
        for a, b in zip(cnt_srs_dev.index, cnt_srs_dev.values):  # Display corresponding number
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
        plt.legend(loc="upper right")
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Ner Label', fontsize=12)
        plt.title("The ner label statistical graph of the dev set")
        # plt.xticks(rotation='vertical')
        plt.show()

    def _is_valid_input_data(self, input_line):
        """is the input data valid"""
        try:
            dic = input_line.strip()
            dic = json.loads(dic)
        except:
            return False
        if "text" not in dic or "postag" not in dic or \
                type(dic["postag"]) is not list:
            return False
        for item in dic['postag']:
            if "word" not in item or "pos" not in item:
                return False
        return True

    def _based_on_sentence_get_token_idx(self, sentence):
        """Get the start offset of every token"""
        token_idx_list = []
        start_idx = 0
        for sent_term in sentence:
            if start_idx >= len(sentence):
                break
            token_idx_list.append(start_idx)
            start_idx += len(sent_term)
        return token_idx_list

    def _get_token_idx(self, sentence_term_list, sentence):
        """Get the start offset of every token"""
        token_idx_list = []
        start_idx = 0
        for sent_term in sentence_term_list:
            if start_idx >= len(sentence):
                break
            token_idx_list.append(start_idx)
            start_idx += len(sent_term)
        return token_idx_list

    def _add_item_offset(self, token, sentence): #'喜剧之王'
        """Get the start and end offset of a token in a sentence"""
        s_pattern = re.compile(re.escape(token), re.I) #re.compile('\\喜\\剧\\之\\王', re.IGNORECASE)
        token_offset_list = []
        for m in s_pattern.finditer(sentence): #m={SRE_MATCH} <_sre.SRE_Match object; span=(21, 25), match='喜剧之王'>
            token_offset_list.append((m.group(), m.start(), m.end()))
        return token_offset_list # <class 'list'>: [('喜剧之王', 21, 25)]

    def _cal_item_pos(self, target_offset, idx_list):
        """Get the index list where the token is located"""
        target_idx = []
        for target in target_offset:
            start, end = target[1], target[2]
            cur_idx = []
            for i, idx in enumerate(idx_list):
                if idx >= start and idx < end:
                    cur_idx.append(i)
            if len(cur_idx) > 0:
                target_idx.append(cur_idx)
        return target_idx

    def _cal_mark_slot(self, spo_list, sentence, token_idx_list, intersection_sub_obj_global):
        """Represent the value of the label"""
        mark_list = ['O'] * len(token_idx_list) #chars per line
        # mark_list = ['O'] * len(sentence)
        sub_temp = set()
        obj_temp = set()
        for spo in spo_list:
            sub = str(spo['subject']).replace(" ", '').replace("《", '').replace("》", '')
            sub_temp.add(sub)
            obj = str(spo['object']).replace(" ", '').replace("《", '').replace("》", '')
            obj_temp.add(obj)
        sub_obj_intersection_line = sub_temp.intersection(obj_temp)
        for sub in sub_temp.difference(sub_obj_intersection_line):
            s_idx_list = self._cal_item_pos(self._add_item_offset(sub, sentence), \
                                            token_idx_list)
            if sub in intersection_sub_obj_global:
                if len(s_idx_list) == 0:
                    continue
                for s_idx in s_idx_list:
                    if len(s_idx) == 1:
                        mark_list[s_idx[0]] = 'B-SUB_OBJ'
                    else:
                        mark_list[s_idx[0]] = 'B-SUB_OBJ'
                        mark_list[s_idx[-1]] = 'I-SUB_OBJ'
                        for idx in range(1, len(s_idx) - 1):
                            mark_list[s_idx[idx]] = 'I-SUB_OBJ'
            else:
                if len(s_idx_list) == 0:
                    continue
                for s_idx in s_idx_list:
                    if len(s_idx) == 1:
                        mark_list[s_idx[0]] = 'B-SUB'
                    else:
                        mark_list[s_idx[0]] = 'B-SUB'
                        mark_list[s_idx[-1]] = 'I-SUB'
                        for idx in range(1, len(s_idx) - 1):
                            mark_list[s_idx[idx]] = 'I-SUB'
        for obj in obj_temp.difference(sub_obj_intersection_line):
            o_idx_list = self._cal_item_pos(self._add_item_offset(obj, sentence), \
                                            token_idx_list)
            if obj in intersection_sub_obj_global:
                if len(o_idx_list) == 0:
                    continue
                for o_idx in o_idx_list:
                    if len(o_idx) == 1:
                        mark_list[o_idx[0]] = 'B-SUB_OBJ'
                    else:
                        mark_list[o_idx[0]] = 'B-SUB_OBJ'
                        mark_list[o_idx[-1]] = 'I-SUB_OBJ'
                        for idx in range(1, len(o_idx) - 1):
                            mark_list[o_idx[idx]] = 'I-SUB_OBJ'
            else:
                if len(o_idx_list) == 0:
                    continue
                for o_idx in o_idx_list:
                    if len(o_idx) == 1:
                        mark_list[o_idx[0]] = 'B-OBJ'
                    else:
                        mark_list[o_idx[0]] = 'B-OBJ'
                        mark_list[o_idx[-1]] = 'I-OBJ'
                        for idx in range(1, len(o_idx) - 1):
                            mark_list[o_idx[idx]] = 'I-OBJ'

        for sub_obj in sub_obj_intersection_line:
            s_o_idx_list = self._cal_item_pos(self._add_item_offset(sub_obj, sentence), \
                                              token_idx_list)
            if len(s_o_idx_list) == 0:
                continue
            for s_o_idx in s_o_idx_list:
                if len(s_o_idx) == 1:
                    mark_list[s_o_idx[0]] = 'B-SUB_OBJ'
                else:
                    mark_list[s_o_idx[0]] = 'B-SUB_OBJ'
                    mark_list[s_o_idx[-1]] = 'I-SUB_OBJ'
                    for idx in range(1, len(s_o_idx) - 1):
                        mark_list[s_o_idx[idx]] = 'I-SUB_OBJ'

        return mark_list

    def _cal_mark_slot_sub(self, spo_list, sentence, token_idx_list):
        """Represent the value of the label"""
        mark_list = ['O'] * len(token_idx_list) #words per line
        for spo in spo_list:
            sub = spo['subject']
            s_idx_list = self._cal_item_pos(self._add_item_offset(sub, sentence), \
                    token_idx_list)
            if len(s_idx_list) == 0:
                continue
            for s_idx in s_idx_list:
                if len(s_idx) == 1:
                    mark_list[s_idx[0]] = 'B-SUB'
                else:
                    mark_list[s_idx[0]] = 'B-SUB'
                    mark_list[s_idx[-1]] = 'I-SUB'
                    for idx in range(1, len(s_idx) - 1):
                        mark_list[s_idx[idx]] = 'I-SUB'
            #print(mark_list)
        return mark_list

    def _cal_mark_slot_obj(self, spo_list, sentence, token_idx_list):
        """Represent the value of the label"""
        mark_list = ['O'] * len(token_idx_list) #words per line
        for spo in spo_list:
            obj = spo['object']
            o_idx_list = self._cal_item_pos(self._add_item_offset(obj, sentence), \
                    token_idx_list)
            if len(o_idx_list) == 0:
                continue
            for o_idx in o_idx_list:
                if len(o_idx) == 1:
                    mark_list[o_idx[0]] = 'B-OBJ'
                else:
                    mark_list[o_idx[0]] = 'B-OBJ'
                    mark_list[o_idx[-1]] = 'I-OBJ'
                    for idx in range(1, len(o_idx) - 1):
                        mark_list[o_idx[idx]] = 'I-OBJ'
            #print(mark_list)
        return mark_list

    def postag_based_on_char(self, word_pos_zip):  # speed is so slow.
        #seg = pkuseg.pkuseg(postag=True)
        #text_seg = seg.cut(text)
        #char_list = []
        postag_list = []
        for item in word_pos_zip:
            word = item[0]
            pos_tag = str(item[1]).upper()
            if len(word) == 1:
                #char_list.append(word)
                postag_list.append(str(pos_tag))
            else:
                char_temp = [c for c in word]
                postag_temp = [pos_tag] * len(char_temp)
                #char_list.extend(char_temp)
                postag_list.extend(postag_temp)
        return postag_list

    def get_sub_obj_bio(self, output_path):
        # There is subject-object overlap, so this approach is wrong.
        # We nend to deal with the subject and object seprately.
        # But the experimental results are not satisfactory.
        # Because We do not consider the intersection of subject and object
        # in the same sentence.
        # Label "B-SUB_OBJ" and "I-SUB_OBJ" in intersecting subjects and objects. 20190423
        # Add postag. 20190424
        logger.info("processing ...")
        intersection_sub_obj_train_set, intersection_sub_obj_dev_set = self._valid_sub_obj_intersection()
        for item in intersection_sub_obj_dev_set:
            if item not in intersection_sub_obj_train_set:
                intersection_sub_obj_train_set.add(item)
        words_all = []
        label_slot_all = []
        postag_all = []
        logger.info("Processing training set")
        #count_sub_obj_overlaped_train = 0
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                #count_overlap_temp = 0
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                dic = line.strip()
                dic = json.loads(dic)
                #sentence = str(dic['text']).replace(" ", '')
                word_list = [item["word"] for item in dic["postag"] if item["word"] != ' ']
                sentence = ''
                for w in word_list:
                    sentence += w
                postag_list = [item["pos"] for item in dic["postag"] if item["word"] != ' ']
                assert "error", len(word_list) != len(postag_list)
                w_p_zip = zip(word_list, postag_list)
                postag_list_ = self.postag_based_on_char(w_p_zip)
                token_idx_list = self._based_on_sentence_get_token_idx(sentence)
                sentence_term_list = [c for c in sentence]
                if 'spo_list' not in dic:
                    label_slot = ['O'] * len(sentence_term_list)
                else:
                    # label_slot, count_overlap_temp_ = self._cal_mark_slot(dic['spo_list'], sentence, token_idx_list,
                    #                                                       count_overlap_temp)
                    label_slot = self._cal_mark_slot(dic['spo_list'], sentence, token_idx_list,
                                                     intersection_sub_obj_train_set)
                    # count_sub_obj_overlaped_train += count_overlap_temp_
                sentence_term_list.append("")
                words_all.extend(sentence_term_list)
                postag_list_.append("")
                postag_all.extend(postag_list_)
                label_slot.append("")
                label_slot_all.extend(label_slot)
        saved_path_home = output_path
        if not os.path.exists(saved_path_home):
            os.makedirs(saved_path_home)
        with open(saved_path_home + 'train.txt', 'w') as output:
                for i, w in enumerate(words_all):
                    output.write(w + ' ' + postag_all[i] + ' ' + label_slot_all[i] + '\n')
                output.close()
        # logger.info("training set has been completed!\nThe same entity in the same sentence appear subject and object cross number: " +
        #             str(count_sub_obj_overlaped_train))  # output:2487
        logger.info("training set has been completed!")

        words_all_dev = []
        label_slot_all_dev = []
        postag_all_dev = []
        logger.info("Processing dev set")
        #count_sub_obj_overlaped_dev = 0
        with open(self.dev_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                #count_overlap_temp = 0
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                dic = line.strip()
                dic = json.loads(dic)
                #sentence = str(dic['text']).replace(" ", '')
                word_list = [item["word"] for item in dic["postag"] if item["word"] != ' ']
                sentence = ''
                for w in word_list:
                    sentence += w
                postag_list = [item["pos"] for item in dic["postag"] if item["word"] != ' ']
                assert "error", len(word_list) != len(postag_list)
                w_p_zip = zip(word_list, postag_list)
                postag_list_ = self.postag_based_on_char(w_p_zip)
                token_idx_list = self._based_on_sentence_get_token_idx(sentence)
                sentence_term_list = [c for c in sentence]
                if 'spo_list' not in dic:
                    label_slot = ['O'] * len(sentence_term_list)
                else:
                    # label_slot, count_overlap_temp_ = self._cal_mark_slot(dic['spo_list'], sentence, token_idx_list,
                    #                                                       count_overlap_temp)
                    label_slot = self._cal_mark_slot(dic['spo_list'], sentence, token_idx_list,
                                                     intersection_sub_obj_train_set)
                    #count_sub_obj_overlaped_dev += count_overlap_temp_
                sentence_term_list.append("")
                words_all_dev.extend(sentence_term_list)
                postag_list_.append("")
                postag_all_dev.extend(postag_list_)
                label_slot.append("")
                label_slot_all_dev.extend(label_slot)

        with open(saved_path_home + 'dev.txt', 'w') as output:
                for i, w in enumerate(words_all_dev):
                    output.write(w + ' ' + postag_all_dev[i] + ' ' + label_slot_all_dev[i] + '\n')
                output.close()
        # logger.info("dev set has been completed!\nThe same entity in the same sentence appear subject and object cross number: "+
        #             str(count_sub_obj_overlaped_dev))  # output:300
        logger.info("dev set has been completed!")

        logger.info("Processing test set")
        words_all_test = []
        label_slot_all_test = []
        postag_all_test = []
        with open(self.test_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
             # count_overlap_temp = 0
             # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                dic = line.strip()
                dic = json.loads(dic)
                #sentence = str(dic['text']).replace(" ", '')
                word_list = [item["word"] for item in dic["postag"] if item["word"] != ' ']
                sentence = ''
                for w in word_list:
                    sentence += w
                postag_list = [item["pos"] for item in dic["postag"] if item["word"] != ' ']
                assert "error", len(word_list) != len(postag_list)
                w_p_zip = zip(word_list, postag_list)
                postag_list_ = self.postag_based_on_char(w_p_zip)
                #token_idx_list = self._based_on_sentence_get_token_idx(sentence)
                sentence_term_list = [c for c in sentence]
                label_slot = ['O'] * len(sentence_term_list)

                sentence_term_list.append("")
                words_all_test.extend(sentence_term_list)
                postag_list_.append("")
                postag_all_test.extend(postag_list_)
                label_slot.append("")
                label_slot_all_test.extend(label_slot)

        with open(saved_path_home + 'test.txt', 'w') as output:
            for i, w in enumerate(words_all_test):
                output.write(w + ' ' + postag_all_test[i] + ' ' + label_slot_all_test[i] + '\n')
            output.close()
        logger.info("test set has been completed!")

    def get_sub_bio(self, ):  ## Just convert the subject of the training set and dev set into bio format.
        logger.info("processing ...")
        words_all = []
        label_slot_all = []
        logger.info("Processing training set")
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                dic = line.strip()
                dic = json.loads(dic)
                sentence = dic['text']  # '如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈'
                sentence_term_list = [item['word'] for item in dic[
                    'postag']]  # <class 'list'>: ['如何', '演', '好', '自己', '的', '角色', '，', '请', '读', '《', '演员自我修养', '》', '《', '喜剧之王', '》', '周星驰', '崛起', '于', '穷困潦倒', '之中', '的', '独门', '秘笈']
                token_idx_list = self._get_token_idx(sentence_term_list,
                                                     sentence)  # <class 'list'>: [0, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 19, 20, 21, 25, 26, 29, 31, 32, 36, 38, 39, 41]
                if 'spo_list' not in dic:
                    label_slot = ['O'] * len(sentence_term_list)
                else:
                    label_slot = self._cal_mark_slot_sub(dic['spo_list'], sentence, token_idx_list)
                sentence_term_list.append(" ")
                words_all.extend(sentence_term_list)
                label_slot.append(" ")
                label_slot_all.extend(label_slot)
        saved_path_home = "./data/BIO_format/subject-bio/"
        if not os.path.exists(saved_path_home):
            os.makedirs(saved_path_home)
        with open(saved_path_home + 'train', 'w') as output:
            for i, w in enumerate(words_all):
                output.write(w + ' ' + label_slot_all[i] + '\n')
            output.close()
        logger.info("training set has been completed!")

        words_all_dev = []
        label_slot_all_dev = []
        logger.info("Processing dev set")
        with open(self.dev_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                dic = line.strip()
                dic = json.loads(dic)
                sentence = dic['text']
                sentence_term_list = [item['word'] for item in dic[
                    'postag']]
                token_idx_list = self._get_token_idx(sentence_term_list,
                                                     sentence)
                if 'spo_list' not in dic:
                    label_slot = ['O'] * len(sentence_term_list)
                else:
                    label_slot = self._cal_mark_slot_sub(dic['spo_list'], sentence, token_idx_list)
                sentence_term_list.append(" ")
                words_all_dev.extend(sentence_term_list)
                label_slot.append(" ")
                label_slot_all_dev.extend(label_slot)
        with open(saved_path_home + 'dev', 'w') as output:
            for i, w in enumerate(words_all_dev):
                output.write(w + ' ' + label_slot_all_dev[i] + '\n')
            output.close()
        logger.info("dev set has been completed!")

    def get_obj_bio(self, ):  ## Just convert the object of the training set and dev set into bio format.
        logger.info("processing ...")
        words_all = []
        #sentence_all = ""
        label_slot_all = []
        logger.info("Processing training set")
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                dic = line.strip()
                dic = json.loads(dic)
                sentence = dic['text']  # '如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈'
                sentence_term_list = [item['word'] for item in dic[
                    'postag']]  # <class 'list'>: ['如何', '演', '好', '自己', '的', '角色', '，', '请', '读', '《', '演员自我修养', '》', '《', '喜剧之王', '》', '周星驰', '崛起', '于', '穷困潦倒', '之中', '的', '独门', '秘笈']
                token_idx_list = self._get_token_idx(sentence_term_list,
                                                     sentence)
                if 'spo_list' not in dic:
                    label_slot = ['O'] * len(sentence_term_list)
                else:
                    label_slot = self._cal_mark_slot_obj(dic['spo_list'], sentence, token_idx_list)
               # sentence = sentence + " "
               # sentence_all += sentence
                sentence_term_list.append(" ")
                words_all.extend(sentence_term_list)
                label_slot.append(" ")
                label_slot_all.extend(label_slot)
                #print(str(len(words_all)), str(len(label_slot_all)))

        saved_path_home = "./data/BIO_format/object-bio/"
        if not os.path.exists(saved_path_home):
            os.makedirs(saved_path_home)
        with open(saved_path_home + 'train.txt', 'w') as output:
            for i, w in enumerate(words_all):
                output.write(w + ' ' + label_slot_all[i] + '\n')
            output.close()
        logger.info("training set has been completed!")

        words_all_dev = []
        #sentence_all_dev = ""
        label_slot_all_dev = []
        logger.info("Processing dev set")
        with open(self.dev_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(line):
                    print('Format is error')
                    return None
                dic = line.strip()
                dic = json.loads(dic)
                sentence = dic['text']
                sentence_term_list = [item['word'] for item in dic[
                    'postag']]
                token_idx_list = self._get_token_idx(sentence_term_list,
                                                     sentence)
                if 'spo_list' not in dic:
                    label_slot = ['O'] * len(sentence_term_list)
                else:
                    label_slot = self._cal_mark_slot_obj(dic['spo_list'], sentence, token_idx_list)
                #sentence = sentence + " "
                #sentence_all_dev += sentence
                sentence_term_list.append(" ")
                words_all_dev.extend(sentence_term_list)
                label_slot.append(" ")
                label_slot_all_dev.extend(label_slot)
        with open(saved_path_home + 'dev.txt', 'w') as output:
            for i, w in enumerate(words_all_dev):
                output.write(w + ' ' + label_slot_all_dev[i] + '\n')
            output.close()
        logger.info("dev set has been completed!")

    def get_character_level_vector_based_word2vector(self,):
        char_list_all = []
        logger.info("process training set")
        with open(self.train_path, "r") as train_f:
            for li in tqdm(train_f):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(li):
                    print('Format is error')
                    return None
                dic = li.strip()
                line = json.loads(dic)
                char = [c for c in line["text"]]
                char_list_all.append(char)
            train_f.close()

        logger.info("process dev set")
        with open(self.dev_path, "r") as dev_f:
            for li in tqdm(dev_f):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(li):
                    print('Format is error')
                    return None
                dic = li.strip()
                line = json.loads(dic)
                char = [c for c in line["text"]]
                char_list_all.append(char)
            dev_f.close()

        # train embedding
        logger.info("Training embedding by using word2vector...")
        uni_model = Word2Vec(char_list_all, size=200, sg=1, window=5, min_count=1, workers=24)
        # uni_model.save("char.vec")
        uni_model.wv.save_word2vec_format('./data/embed/char_word2vec.vec', binary=False)
        logger.info("All finished! char_embed are saved at: ./data/embed/char_word2vec.vec")

    def convert_data_format_for_RE_BGRU_2ATT(self, output_train_path, output_dev_path):

        logger.info("process training set")
        with open(output_train_path, "w") as output_train:
            with open(self.train_path, "r") as train_f:
                for li in tqdm(train_f):
                    # verify that the input format of each line meets the format
                    if not self._is_valid_input_data(li):
                        print('Format is error')
                        return None
                    dic = li.strip()
                    dic = json.loads(dic)
                    text = dic["text"]
                    for spo in dic["spo_list"]:
                        sub = spo["subject"]
                        obj = spo["object"]
                        p = spo["predicate"]
                        output_train.write(sub + "\t" + obj + "\t" + p + "\t" + text + '\n')
                train_f.close()
            output_train.close()

        logger.info("process set set")
        with open(output_dev_path, "w") as output_dev:
            with open(self.dev_path, "r") as dev_f:
                for li in tqdm(dev_f):
                    # verify that the input format of each line meets the format
                    if not self._is_valid_input_data(li):
                        print('Format is error')
                        return None
                    dic = li.strip()
                    dic = json.loads(dic)
                    text = dic["text"]
                    for spo in dic["spo_list"]:
                        sub = spo["subject"]
                        obj = spo["object"]
                        p = spo["predicate"]
                        output_dev.write(sub + "\t" + obj + "\t" + p + "\t" + text + '\n')
                dev_f.close()
            output_dev.close()
            logger.info("All complete, output files are saved at: " + output_train_path + " " + "and" + " " + output_dev_path)

    def _load_p_eng_dict(self, dict_name):
        """load label dict from file"""
        p_eng_dict = {}
        with open(dict_name, 'r') as fr:
            for idx, line in enumerate(fr):
                p, p_eng = line.strip().split('\t')
                p_eng_dict[p] = p_eng
        return p_eng_dict


    ###Deal with participle less cutting problem
    #raw text: 参考文献[1]彭兰，《网络传播概论》，第三版，中国人民大学出版社，261页
    #After jieba: "参考文献|[|1|]|彭兰|，|《|网络传播概论|》|，|第三版|，|中国人民大学出版社|，|261|页"
    #Except output: "参考文献|[|1|]|彭兰|，|《|网络传播概论|》|，|第三版|，|中国|人民大学|出版社|，|261|页"
    #网络传播概论    人民大学
    def deal_with_participle_less_cutting_problem(self, text, sub, obj):
        jieba.add_word(sub)
        jieba.add_word(obj)
        text = text.replace(" ", "")
        text_just_jieba = " ".join(jieba.cut(text))
        #print("After jieba seg:\n" + text)
        text_seg_list = text_just_jieba.split(" ")

        if sub not in text_seg_list and obj not in text_seg_list:
            # print("sorry, after jieba, our problems are all unresolved")
            # sent_updated = ''
            # for w in text_seg_list:
            #     sent_updated += w
            # sent_updated = sent_updated.strip()
            words = []
            for w in text_seg_list:
                if sub in w and obj not in w:
                    sub_char = [c for c in sub]
                    w_char = [c for c in w]
                    temp_start_end_total = []
                    temp = []
                    for i, w_ in enumerate(w_char):
                        if w_ == sub_char[0]:
                            temp.append(i)
                        if w_ == sub_char[-1]:
                            temp.append(i)
                            temp_start_end_total.append(temp)
                            temp = []
                    for start_end in temp_start_end_total:
                        if len(start_end) == 1:
                            start, end = start_end[0], start_end[-1]
                            w_char.insert(start, "|")
                            w_char.insert(end + 2, "|")
                        if len(start_end) > 1:
                            start, end = start_end[0], start_end[1]
                            w_char.insert(start, "|")
                            w_char.insert(end + 2, "|")
                    sent_temp = ''
                    for c in w_char:
                        sent_temp += c
                    sent_temp = sent_temp.strip().split("|")
                    for w in sent_temp:
                        words.append(w)

                if obj in w and sub not in w:
                    obj_char = [c for c in obj]
                    w_char = [c for c in w]
                    temp_start_end_total = []
                    temp = []
                    for i, w_ in enumerate(w_char):
                        if w_ == obj_char[0]:
                            temp.append(i)
                        if w_ == obj_char[-1]:
                            temp.append(i)
                            temp_start_end_total.append(temp)
                            temp = []
                    for start_end in temp_start_end_total:
                        if len(start_end) == 1:
                            start, end = start_end[0], start_end[-1]
                            w_char.insert(start, "|")
                            w_char.insert(end + 2, "|")
                        if len(start_end) > 1:
                            start, end = start_end[0], start_end[1]
                            w_char.insert(start, "|")
                            w_char.insert(end + 2, "|")
                    sent_temp = ''
                    for c in w_char:
                        sent_temp += c
                    sent_temp = sent_temp.strip().split("|")
                    # sent_temp2 = ''
                    # for w in sent_temp:
                    #     sent_temp2 += w + " "
                    # sent_temp2 = sent_temp2.strip()
                    for w in sent_temp:
                        words.append(w)
                    # words.append(sent_temp[])
                    # print(sent_temp2)
                    # print(str(temp_start_end_total))
                else:
                    words.append(w)

            sent_updated = ''
            for w in words:
                sent_updated += w
            sent_updated = sent_updated.strip()

            return sent_updated, words

        #Start adding a rule to process it.
        elif sub in text_seg_list and obj not in text_seg_list:
            words = []
            for w in text_seg_list:
                if obj in w:
                    obj_char = [c for c in obj]
                    w_char = [c for c in w]
                    temp_start_end_total = []
                    temp = []
                    for i, w_ in enumerate(w_char):
                        if w_ == obj_char[0]:
                            temp.append(i)
                        if w_ == obj_char[-1]:
                            temp.append(i)
                            temp_start_end_total.append(temp)
                            temp = []
                    for start_end in temp_start_end_total:
                        if len(start_end) == 1:
                            start, end = start_end[0], start_end[-1]
                            w_char.insert(start, "|")
                            w_char.insert(end + 2, "|")
                        if len(start_end) > 1:
                            start, end = start_end[0], start_end[1]
                            w_char.insert(start, "|")
                            w_char.insert(end + 2, "|")

                    sent_temp = ''
                    for c in w_char:
                        sent_temp += c
                    sent_temp = sent_temp.strip().split("|")
                    # sent_temp2 = ''
                    # for w in sent_temp:
                    #     sent_temp2 += w + " "
                    # sent_temp2 = sent_temp2.strip()
                    for w in sent_temp:
                        words.append(w)
                    # words.append(sent_temp[])
                    # print(sent_temp2)
                    # print(str(temp_start_end_total))
                else:
                    words.append(w)

            sent_updated = ''
            for w in words:
                sent_updated += w
            sent_updated = sent_updated.strip()
            #print("Custom rule optimized word segmentation results: \n" + sent_updated)
            return sent_updated, words

        elif sub not in text_seg_list and obj in text_seg_list:
            words = []
            for w in text_seg_list:
                if sub in w:
                    obj_char = [c for c in sub]
                    w_char = [c for c in w]
                    temp_start_end_total = []
                    temp = []
                    for i, w_ in enumerate(w_char):
                        if w_ == obj_char[0]:
                            temp.append(i)
                        if w_ == obj_char[-1]:
                            temp.append(i)
                            temp_start_end_total.append(temp)
                            temp = []
                    for start_end in temp_start_end_total:
                        if len(start_end) == 1:
                            start, end = start_end[0], start_end[-1]
                            w_char.insert(start, "|")
                            w_char.insert(end + 2, "|")
                        if len(start_end) > 1:
                            start, end = start_end[0], start_end[1]
                            w_char.insert(start, "|")
                            w_char.insert(end + 2, "|")
                    sent_temp = ''
                    for c in w_char:
                        sent_temp += c
                    sent_temp = sent_temp.strip().split("|")
                    # sent_temp2 = ''
                    # for w in sent_temp:
                    #     sent_temp2 += w + " "
                    # sent_temp2 = sent_temp2.strip()
                    for w in sent_temp:
                        words.append(w)
                    # words.append(sent_temp)
                    # print(sent_temp2)
                    # print(str(temp_start_end_total))
                else:
                    words.append(w)
            sent_updated = ''
            for w in words:
                sent_updated += w
            sent_updated = sent_updated.strip()
            #print("Custom rule optimized word segmentation results: \n" + sent_updated)
            return sent_updated, words

        elif sub in text_seg_list and obj in text_seg_list:
            #print("Congratulations! Just use jieba all solve our problems.")
            sent_updated = ''
            for w in text_seg_list:
                sent_updated += w
            sent_updated = sent_updated.strip()
            return sent_updated, text_seg_list

    def convert_data_format_for_entity_aware_relation_classification_1(self, p_eng_dict_path, output_train_path, output_dev_path):

        self._p_map_eng_dict = self._load_p_eng_dict(p_eng_dict_path)
        logger.info("process training set")

        with open(output_train_path, "w") as output_train:
            with open(self.train_path, "r") as train_f:
                count_error_train = 0
                count_error_handled_train = 0
                line_id_train = 0
                for li in tqdm(train_f):
                    # verify that the input format of each line meets the format
                    if not self._is_valid_input_data(li):
                        print('Format is error')
                        return None
                    dic = li.strip()
                    dic = json.loads(dic)
                    sentence = dic['text']
                    sentence_term_list = [item['word'] for item in dic['postag']]
                    token_idx_list = self._get_token_idx(sentence_term_list, sentence)
                    if not dic["spo_list"]:
                        continue
                    for spo in dic["spo_list"]:
                        sub = spo['subject']
                        obj = spo['object']
                        relation = self._p_map_eng_dict.get(spo["predicate"])
                        s_idx_list = self._cal_item_pos(self._add_item_offset(sub, sentence),\
                                                            token_idx_list)            ##[[0,1,2]]
                        try:
                            sentence_term_list_copy = sentence_term_list.copy()
                            sentence_term_list_copy.insert(s_idx_list[0][0], "<subject>")
                            sentence_term_list_copy.insert(s_idx_list[0][-1] + 2, "</subject>")
                            sent_line = ''
                            for w in sentence_term_list_copy:
                                sent_line += w
                            sent_line = sent_line.strip()

                            token_idx_list_ = self._get_token_idx(sentence_term_list_copy, sent_line)
                            o_idx_list = self._cal_item_pos(self._add_item_offset(obj, sent_line), \
                                                            token_idx_list_)
                            sentence_term_list_copy.insert(o_idx_list[0][0], "<object>")
                            sentence_term_list_copy.insert(o_idx_list[0][-1] + 2, "</object>")

                            sent_output = ''
                            for w in sentence_term_list_copy:
                                sent_output += w + ' '
                            sent_line_update = sent_output.strip()
                            output_train.write(str(line_id_train) + "\t" + sent_line_update + "\n")
                            output_train.write(relation + "(subject,object)" + "\n\n")
                            line_id_train += 1
                        except:    ##Optimize the participle according to the subject and object
                            count_error_train += 1
                            jieba.add_word(sub)
                            jieba.add_word(obj)
                            sent_seg = "|".join(jieba.cut(sentence))
                            jieba.del_word(sub)
                            jieba.del_word(obj)
                            #jieba.suggest_freq()
                            words_seg_list = sent_seg.split("|")
                            sent_output_ = ''
                            for w in words_seg_list:
                                sent_output_ += w
                            sent_line_updated = sent_output_.strip()
                            token_idx_list_ = self._get_token_idx(words_seg_list, sent_line_updated)
                            s_idx_list_ = self._cal_item_pos(self._add_item_offset(sub, sent_line_updated), \
                                                            token_idx_list_)
                            try:
                                words_seg_list_copy = words_seg_list.copy()
                                words_seg_list_copy.insert(s_idx_list_[0][0], "<subject>")
                                words_seg_list_copy.insert(s_idx_list_[0][-1] + 2, "</subject>")
                                sent_line = ''
                                for w in words_seg_list_copy:
                                    sent_line += w
                                sent_line = sent_line.strip()

                                token_idx_list_ = self._get_token_idx(words_seg_list_copy, sent_line)
                                o_idx_list_ = self._cal_item_pos(self._add_item_offset(obj, sent_line), \
                                                                token_idx_list_)
                                words_seg_list_copy.insert(o_idx_list_[0][0], "<object>")
                                words_seg_list_copy.insert(o_idx_list_[0][-1] + 2, "</object>")

                                sent_output = ''
                                for w in words_seg_list_copy:
                                    sent_output += w + ' '
                                sent_line_updated = sent_output.strip()
                                output_train.write(str(line_id_train) + "\t" + sent_line_updated + "\n")
                                output_train.write(relation + "(subject,object)" + "\n\n")
                                line_id_train += 1
                                count_error_handled_train += 1
                            except:
                                   print("ERROR")
                                   print(sentence + "\n" + sent_seg + "\n" + sub + "\t" + obj + "\n")
            output_train.close()

            logger.info("process dev set....")
            with open(output_dev_path, "w") as output_dev:
                with open(self.dev_path, "r") as dev_f:
                    count_error_dev = 0
                    count_error_handled_dev = 0
                    line_id_dev = 0
                    for li in tqdm(dev_f):
                        # verify that the input format of each line meets the format
                        if not self._is_valid_input_data(li):
                            print('Format is error')
                            return None
                        dic = li.strip()
                        dic = json.loads(dic)
                        sentence = dic['text']
                        sentence_term_list = [item['word'] for item in dic['postag']]
                        token_idx_list = self._get_token_idx(sentence_term_list, sentence)
                        if not dic["spo_list"]:
                            continue
                        for spo in dic["spo_list"]:
                            sub = spo['subject']
                            obj = spo['object']
                            relation = self._p_map_eng_dict.get(spo["predicate"])
                            s_idx_list = self._cal_item_pos(self._add_item_offset(sub, sentence),\
                                                                token_idx_list)            ##[[0,1,2]]
                            try:
                                sentence_term_list_copy = sentence_term_list.copy()
                                sentence_term_list_copy.insert(s_idx_list[0][0], "<subject>")
                                sentence_term_list_copy.insert(s_idx_list[0][-1] + 2, "</subject>")
                                sent_line = ''
                                for w in sentence_term_list_copy:
                                    sent_line += w
                                sent_line = sent_line.strip()

                                token_idx_list_ = self._get_token_idx(sentence_term_list_copy, sent_line)
                                o_idx_list = self._cal_item_pos(self._add_item_offset(obj, sent_line), \
                                                                token_idx_list_)
                                sentence_term_list_copy.insert(o_idx_list[0][0], "<object>")
                                sentence_term_list_copy.insert(o_idx_list[0][-1] + 2, "</object>")

                                sent_output = ''
                                for w in sentence_term_list_copy:
                                    sent_output += w + ' '
                                sent_line_update = sent_output.strip()
                                output_dev.write(str(line_id_dev) + "\t" + sent_line_update + "\n")
                                output_dev.write(relation + "(subject,object)" + "\n\n")
                                line_id_dev += 1
                            except:    ##Optimize the participle according to the subject and object
                                count_error_dev += 1
                                jieba.add_word(sub)
                                jieba.add_word(obj)
                                sent_seg = "|".join(jieba.cut(sentence))
                                jieba.del_word(sub)
                                jieba.del_word(obj)
                                words_seg_list = sent_seg.split("|")
                                sent_output_ = ''
                                for w in words_seg_list:
                                    sent_output_ += w
                                sent_line_updated = sent_output_.strip()
                                token_idx_list_ = self._get_token_idx(words_seg_list, sent_line_updated)
                                s_idx_list_ = self._cal_item_pos(self._add_item_offset(sub, sent_line_updated), \
                                                                token_idx_list_)
                                try:
                                    words_seg_list_copy = words_seg_list.copy()
                                    words_seg_list_copy.insert(s_idx_list_[0][0], "<subject>")
                                    words_seg_list_copy.insert(s_idx_list_[0][-1] + 2, "</subject>")
                                    sent_line = ''
                                    for w in words_seg_list_copy:
                                        sent_line += w
                                    sent_line = sent_line.strip()

                                    token_idx_list_ = self._get_token_idx(words_seg_list_copy, sent_line)
                                    o_idx_list_ = self._cal_item_pos(self._add_item_offset(obj, sent_line), \
                                                                    token_idx_list_)
                                    words_seg_list_copy.insert(o_idx_list_[0][0], "<object>")
                                    words_seg_list_copy.insert(o_idx_list_[0][-1] + 2, "</object>")

                                    sent_output = ''
                                    for w in words_seg_list_copy:
                                        sent_output += w + ' '
                                    sent_line_updated = sent_output.strip()
                                    output_dev.write(str(line_id_dev) + "\t" + sent_line_updated + "\n")
                                    output_dev.write(relation + "(subject,object)" + "\n\n")
                                    line_id_dev += 1
                                    count_error_handled_dev += 1
                                except:
                                       print("ERROR")
                                       print(sentence + "\n" + sent_seg + "\n" + sub + "\t" + obj + "\n")

                print("Wrong number of sentences in training set: " + str(count_error_train) + "\n" + \
                      "Number of sentences processed in training set: " + str(count_error_handled_train) + "\n" + \
                      "Number of unresolved sentences in training set: " + str(
                    count_error_train - count_error_handled_train) + "\n")

                print("Wrong number of sentences in dev set: " + str(count_error_dev) + "\n" + \
                      "Number of sentences processed in dev set: " + str(count_error_handled_dev) + "\n" + \
                      "Number of unresolved sentences in dev set: " + str(count_error_dev - count_error_handled_dev))

            output_dev.close()
            ####The following is a sample output:
            # 0	如何 演 好 自己 的 角色 ， 请 读 《 演员自我修养 》 《 <subject> 喜剧之王 </subject> 》 <object> 周星驰 </object> 崛起 于 穷困潦倒 之中 的 独门 秘笈
            # ACT(subject,object)

            # 1	<subject> 茶树茶网蝽 </subject> ， Stephanitis chinensis Drake ， 属 <object> 半翅目 </object> 网蝽科冠网椿属 的 一种 昆虫
            # BO(subject,object)

    ###On the basis of method 1, join the rules and lower badcase.
    def convert_data_format_for_entity_aware_relation_classification_2(self,
                                                                       p_eng_dict_path,
                                                                       output_train_path,
                                                                       output_dev_path,
                                                                       output_test_path,
                                                                       output_bad_case_path):

        self._p_map_eng_dict = self._load_p_eng_dict(p_eng_dict_path)
        logger.info("process training set")

        with open(output_train_path, "w") as output_train:
            with open(self.train_path, "r") as train_f:
                count_error_train = 0
                count_error_handled_train = 0
                row_id_train = 1
                for li in tqdm(train_f.readlines()):
                    # verify that the input format of each line meets the format
                    if not self._is_valid_input_data(li):
                        print('Format is error')
                        return None
                    dic = li.strip()
                    dic = json.loads(dic)
                    sentence = dic['text']
                    sentence = str(sentence).replace(" ", '').lower()
                    sentence_term_list = [str(item['word']).lower() for item in dic['postag'] if item["word"] != " "]
                    token_idx_list = self._get_token_idx(sentence_term_list, sentence)
                    if not dic["spo_list"]:
                        row_id_train += 1
                        continue
                    for spo in dic["spo_list"]:
                        sub = str(spo['subject']).replace(" ", '').lower()
                        obj = str(spo['object']).replace(" ", '').lower()
                        relation = self._p_map_eng_dict.get(str(spo["predicate"]).replace(" ", ''))
                        s_idx_list = self._cal_item_pos(self._add_item_offset(sub, sentence),\
                                                            token_idx_list)            ##[[0,1,2]]
                        try:
                            sentence_term_list_copy = sentence_term_list.copy()
                            sentence_term_list_copy.insert(s_idx_list[0][0], "<subject>")
                            sentence_term_list_copy.insert(s_idx_list[0][-1] + 2, "</subject>")
                            sent_line = ''
                            for w in sentence_term_list_copy:
                                sent_line += w
                            sent_line = sent_line.strip()

                            token_idx_list_ = self._get_token_idx(sentence_term_list_copy, sent_line)
                            o_idx_list = self._cal_item_pos(self._add_item_offset(obj, sent_line), \
                                                            token_idx_list_)
                            sentence_term_list_copy.insert(o_idx_list[0][0], "<object>")
                            sentence_term_list_copy.insert(o_idx_list[0][-1] + 2, "</object>")

                            sent_output = ''
                            for w in sentence_term_list_copy:
                                sent_output += w + ' '
                            sent_line_update = sent_output.strip()
                            output_train.write(str(row_id_train) + "\t" + sent_line_update + "\n")
                            output_train.write(relation + "(subject,object)" + '\t' + sub + '\t' + obj + "\n\n")
                        except:    ##Optimize the participle according to the subject and object
                            count_error_train += 1
                            sent_line_updated, words_seg_updated = self.deal_with_participle_less_cutting_problem(sentence, sub, obj)

                            token_idx_list_ = self._get_token_idx(words_seg_updated, sent_line_updated)
                            s_idx_list_ = self._cal_item_pos(self._add_item_offset(sub, sent_line_updated), \
                                                            token_idx_list_)
                            try:
                                words_seg_updated_copy = words_seg_updated.copy()
                                words_seg_updated_copy.insert(s_idx_list_[0][0], "<subject>")
                                words_seg_updated_copy.insert(s_idx_list_[0][-1] + 2, "</subject>")
                                sent_line = ''
                                for w in words_seg_updated_copy:
                                    sent_line += w
                                sent_line = sent_line.strip()

                                token_idx_list_ = self._get_token_idx(words_seg_updated_copy, sent_line)
                                o_idx_list_ = self._cal_item_pos(self._add_item_offset(obj, sent_line), \
                                                                token_idx_list_)
                                words_seg_updated_copy.insert(o_idx_list_[0][0], "<object>")
                                words_seg_updated_copy.insert(o_idx_list_[0][-1] + 2, "</object>")

                                sent_output = ''
                                for w in words_seg_updated_copy:
                                    sent_output += w + ' '
                                sent_line_updated = sent_output.strip()
                                output_train.write(str(row_id_train) + "\t" + sent_line_updated + "\n")
                                output_train.write(relation + "(subject,object)" + '\t' + sub + '\t' + obj + "\n\n")

                                count_error_handled_train += 1

                            except:
                                print("Error, unsolved")
                                with open(output_bad_case_path, "a+") as bad_case:
                                    bad_case.write("bad case-training set\n")
                                    log_f = sentence + "\n" + str(
                                        words_seg_updated) + "\n" + sub + "\t" + obj + "\t" + relation + "\n"
                                    bad_case.write(log_f)
                                    #print(log_f)
                                bad_case.close()
                    row_id_train += 1

            output_train.close()

        logger.info("process dev set....")
        with open(output_dev_path, "w") as output_dev:
            with open(self.dev_path, "r") as dev_f:
                count_error_dev = 0
                count_error_handled_dev = 0
                row_id_dev = 1
                for li in tqdm(dev_f.readlines()):
                    # verify that the input format of each line meets the format
                    if not self._is_valid_input_data(li):
                        print('Format is error')
                        return None
                    dic = li.strip()
                    dic = json.loads(dic)
                    sentence = dic['text']
                    sentence = str(sentence).replace(" ", '').lower()
                    sentence_term_list = [str(item['word']).lower() for item in dic['postag'] if item["word"] != " "]
                    token_idx_list = self._get_token_idx(sentence_term_list, sentence)
                    if not dic["spo_list"]:
                        row_id_dev += 1
                        continue
                    for spo in dic["spo_list"]:
                        sub = str(spo['subject']).replace(" ", '').lower()
                        obj = str(spo['object']).replace(" ", '').lower()
                        relation = self._p_map_eng_dict.get(str(spo["predicate"]).replace(" ", ''))
                        s_idx_list = self._cal_item_pos(self._add_item_offset(sub, sentence),\
                                                            token_idx_list)            ##[[0,1,2]]
                        try:
                            sentence_term_list_copy = sentence_term_list.copy()
                            sentence_term_list_copy.insert(s_idx_list[0][0], "<subject>")
                            sentence_term_list_copy.insert(s_idx_list[0][-1] + 2, "</subject>")
                            sent_line = ''
                            for w in sentence_term_list_copy:
                                sent_line += w
                            sent_line = sent_line.strip()

                            token_idx_list_ = self._get_token_idx(sentence_term_list_copy, sent_line)
                            o_idx_list = self._cal_item_pos(self._add_item_offset(obj, sent_line), \
                                                            token_idx_list_)
                            sentence_term_list_copy.insert(o_idx_list[0][0], "<object>")
                            sentence_term_list_copy.insert(o_idx_list[0][-1] + 2, "</object>")

                            sent_output = ''
                            for w in sentence_term_list_copy:
                                sent_output += w + ' '
                            sent_line_update = sent_output.strip()
                            output_dev.write(str(row_id_dev) + "\t" + sent_line_update + "\n")
                            output_dev.write(relation + "(subject,object)" + '\t' + sub + '\t' + obj + "\n\n")
                        except:    ##Optimize the participle according to the subject and object
                            count_error_dev += 1
                            sent_line_updated, words_seg_updated = self.deal_with_participle_less_cutting_problem(sentence, sub, obj)
                            token_idx_list_ = self._get_token_idx(words_seg_updated, sent_line_updated)
                            s_idx_list_ = self._cal_item_pos(self._add_item_offset(sub, sent_line_updated), \
                                                            token_idx_list_)
                            try:
                                words_seg_updated_copy = words_seg_updated.copy()
                                words_seg_updated_copy.insert(s_idx_list_[0][0], "<subject>")
                                words_seg_updated_copy.insert(s_idx_list_[0][-1] + 2, "</subject>")
                                sent_line = ''
                                for w in words_seg_updated_copy:
                                    sent_line += w
                                sent_line = sent_line.strip()

                                token_idx_list_ = self._get_token_idx(words_seg_updated_copy, sent_line)
                                o_idx_list_ = self._cal_item_pos(self._add_item_offset(obj, sent_line), \
                                                                token_idx_list_)
                                words_seg_updated_copy.insert(o_idx_list_[0][0], "<object>")
                                words_seg_updated_copy.insert(o_idx_list_[0][-1] + 2, "</object>")

                                sent_output = ''
                                for w in words_seg_updated_copy:
                                    sent_output += w + ' '
                                sent_line_updated = sent_output.strip()
                                output_dev.write(str(row_id_dev) + "\t" + sent_line_updated + "\n")
                                output_dev.write(relation + "(subject,object)" + '\t' + sub + '\t' + obj + "\n\n")
                                count_error_handled_dev += 1
                            except:
                                print("Error, unsolved")
                                with open(output_bad_case_path, "a+") as bad_case:
                                    bad_case.write("bad case-dev set\n")
                                    log_f = sentence + "\n" + str(
                                        words_seg_updated) + "\n" + sub + "\t" + obj + "\t" + relation + "\n"
                                    bad_case.write(log_f)
                                    #print(log_f)
                                bad_case.close()
                    row_id_dev += 1
            output_dev.close()

        logger.info("process test set....")
        with open(output_test_path, "w") as output_test:
            with open(self.test_path, "r") as test_f:
                count_error_test = 0
                count_error_handled_test = 0
                row_id_test = 1
                for li in tqdm(test_f.readlines()):
                    # verify that the input format of each line meets the format
                    if not self._is_valid_input_data(li):
                        print('Format is error')
                        return None
                    dic = li.strip()
                    dic = json.loads(dic)
                    sentence = dic['text']
                    sentence = str(sentence).replace(" ", '').lower()
                    sentence_term_list = [str(item['word']).lower() for item in dic['postag'] if
                                          item["word"] != " "]
                    token_idx_list = self._get_token_idx(sentence_term_list, sentence)
                    if not dic["spo_list"]:
                        row_id_test += 1
                        continue
                    for spo in dic["spo_list"]:
                        sub = str(spo['subject']).replace(" ", '').lower()
                        obj = str(spo['object']).replace(" ", '').lower()
                        # relation = self._p_map_eng_dict.get(str(spo["predicate"]).replace(" ", ''))
                        relation = 'ACT'
                        s_idx_list = self._cal_item_pos(self._add_item_offset(sub, sentence), \
                                                        token_idx_list)  ##[[0,1,2]]
                        try:
                            sentence_term_list_copy = sentence_term_list.copy()
                            sentence_term_list_copy.insert(s_idx_list[0][0], "<subject>")
                            sentence_term_list_copy.insert(s_idx_list[0][-1] + 2, "</subject>")
                            sent_line = ''
                            for w in sentence_term_list_copy:
                                sent_line += w
                            sent_line = sent_line.strip()

                            token_idx_list_ = self._get_token_idx(sentence_term_list_copy, sent_line)
                            o_idx_list = self._cal_item_pos(self._add_item_offset(obj, sent_line), \
                                                            token_idx_list_)
                            sentence_term_list_copy.insert(o_idx_list[0][0], "<object>")
                            sentence_term_list_copy.insert(o_idx_list[0][-1] + 2, "</object>")

                            sent_output = ''
                            for w in sentence_term_list_copy:
                                sent_output += w + ' '
                            sent_line_update = sent_output.strip()
                            output_test.write(str(row_id_test) + "\t" + sent_line_update + "\n")
                            output_test.write(relation + "(subject,object)" + '\t' + sub + '\t' + obj + "\n\n")
                            #line_id_test += 1
                        except:  ##Optimize the participle according to the subject and object
                            count_error_test += 1
                            sent_line_updated, words_seg_updated = self.deal_with_participle_less_cutting_problem(
                                sentence, sub, obj)
                            token_idx_list_ = self._get_token_idx(words_seg_updated, sent_line_updated)
                            s_idx_list_ = self._cal_item_pos(self._add_item_offset(sub, sent_line_updated), \
                                                             token_idx_list_)
                            try:
                                words_seg_updated_copy = words_seg_updated.copy()
                                words_seg_updated_copy.insert(s_idx_list_[0][0], "<subject>")
                                words_seg_updated_copy.insert(s_idx_list_[0][-1] + 2, "</subject>")
                                sent_line = ''
                                for w in words_seg_updated_copy:
                                    sent_line += w
                                sent_line = sent_line.strip()

                                token_idx_list_ = self._get_token_idx(words_seg_updated_copy, sent_line)
                                o_idx_list_ = self._cal_item_pos(self._add_item_offset(obj, sent_line), \
                                                                 token_idx_list_)
                                words_seg_updated_copy.insert(o_idx_list_[0][0], "<object>")
                                words_seg_updated_copy.insert(o_idx_list_[0][-1] + 2, "</object>")

                                sent_output = ''
                                for w in words_seg_updated_copy:
                                    sent_output += w + ' '
                                sent_line_updated = sent_output.strip()
                                output_test.write(str(row_id_test) + "\t" + sent_line_updated + "\n")
                                output_test.write(relation + "(subject,object)" + '\t' + sub + '\t' + obj + "\n\n")
                                #line_id_test += 1
                                count_error_handled_test += 1
                            except:
                                print("Error, unsolved")
                                with open(output_bad_case_path, "a+") as bad_case:
                                    bad_case.write("bad case-test set\n")
                                    log_f = sentence + "\n" + str(
                                        words_seg_updated) + "\n" + sub + "\t" + obj + "\t" + relation + "\n"
                                    bad_case.write(log_f)
                                    # print(log_f)
                                bad_case.close()
                    row_id_test += 1

                with open(output_bad_case_path, "a+") as bad_case:

                    log_f_train = "Wrong number of sentences in training set: " + str(count_error_train) + "\n" + \
                      "Number of sovled sentences in training set: " + str(count_error_handled_train) + "\n" + \
                      "Number of unresolved sentences in training set: " + str(
                    count_error_train - count_error_handled_train) + "\n"
                    bad_case.write("\n\n")
                    bad_case.write(log_f_train)
                    print(log_f_train)

                    log_f_dev = "Wrong number of sentences in dev set: " + str(count_error_dev) + "\n" + \
                      "Number of solved sentences in dev set: " + str(count_error_handled_dev) + "\n" + \
                      "Number of unresolved sentences in dev set: " + str(count_error_dev - count_error_handled_dev) + \
                                '\n'
                    bad_case.write(log_f_dev + '\n\n')
                    print(log_f_dev)

                    log_f_test = "Wrong number of sentences in test set: " + str(count_error_test) + "\n" + \
                                "Number of solved sentences in test set: " + str(count_error_handled_test) + "\n" + \
                                "Number of unresolved sentences in test set: " + str(
                        count_error_test - count_error_handled_test)
                    bad_case.write(log_f_test)
                    print(log_f_test)
                bad_case.close()


            ####The following is a sample output:
            # 0	如何 演 好 自己 的 角色 ， 请 读 《 演员自我修养 》 《 <subject> 喜剧之王 </subject> 》 <object> 周星驰 </object> 崛起 于 穷困潦倒 之中 的 独门 秘笈
            # ACT(subject,object)

            # 1	<subject> 茶树茶网蝽 </subject> ， Stephanitis chinensis Drake ， 属 <object> 半翅目 </object> 网蝽科冠网椿属 的 一种 昆虫
            # BO(subject,object)

    # The method do not add subject and object markers, and add "other" relationship,
    # later comparative experimental results.
    def convert_data_format_for_entity_aware_relation_classification_3(self,
                                                                       p_eng_dict_path,
                                                                       output_train_path,
                                                                       output_dev_path,
                                                                       output_bad_case_path):

        self._p_map_eng_dict = self._load_p_eng_dict(p_eng_dict_path)
        logger.info("process training set")

        with open(output_train_path, "w") as output_train:
            output_train.write("id" + "\t" + "text" + "\t" + "text_seg" + "\t" +
                               "subject" + "\t" + "subject_start_end" + "\t" +
                               "object" + "\t" + "object_start_end" + "\n")  # text header
            with open(self.train_path, "r") as train_f:
                count_error_train = 0
                count_error_handled_train = 0
                line_id_train = 0
                for li in tqdm(train_f.readlines()):
                    # verify that the input format of each line meets the format
                    if not self._is_valid_input_data(li):
                        print('Format is error')
                        return None
                    dic = li.strip()
                    dic = json.loads(dic)
                    sentence = dic['text']
                    sentence = str(sentence).replace(" ", '').lower()
                    sentence_term_list = [str(item['word']).lower() for item in dic['postag']
                                          if item["word"] != " "]
                    token_idx_list = self._get_token_idx(sentence_term_list, sentence)
                    if not dic["spo_list"]:
                        continue
                    for spo in dic["spo_list"]:
                        sub = str(spo['subject']).replace(" ", '').lower()
                        obj = str(spo['object']).replace(" ", '').lower()
                        relation = self._p_map_eng_dict.get(str(spo["predicate"]).replace(" ", ''))
                        s_idx_list = self._cal_item_pos(self._add_item_offset(sub, sentence),
                                                        token_idx_list)            # [[0,1,2]]
                        o_idx_list = self._cal_item_pos(self._add_item_offset(obj, sentence),
                                                        token_idx_list)
                        try:
                            subject_start_end = str(s_idx_list[0][0]) + "\t" + str(s_idx_list[0][-1])
                            object_start_end = str(o_idx_list[0][0]) + "\t" + str(o_idx_list[0][-1])

                            sent_seg = ''
                            for w in sentence_term_list:
                                sent_seg += w + ' '
                            sent_seg = sent_seg.strip()
                            output_train.write(str(line_id_train) + "\t" + "\"" + sentence + "\"" +
                                               "\t" + "\"" + sent_seg + "\"" + "\t" + sub + "\t" +
                                               subject_start_end + "\t" + obj + "\t" + object_start_end + "\n")
                            output_train.write(relation + "(subject,object)" + "\n\n")
                            line_id_train += 1
                        except:    ##Optimize the participle according to the subject and object
                            count_error_train += 1
                            sent_line_updated, words_seg_updated = self.deal_with_participle_less_cutting_problem(sentence, sub, obj)
                            token_idx_list_ = self._get_token_idx(words_seg_updated, sent_line_updated)
                            s_idx_list_ = self._cal_item_pos(self._add_item_offset(sub, sent_line_updated), \
                                                            token_idx_list_)
                            o_idx_list_ = self._cal_item_pos(self._add_item_offset(obj, sent_line_updated), \
                                                             token_idx_list_)
                            try:
                                subject_start_end_ = str(s_idx_list_[0][0]) + "\t" + str(s_idx_list_[0][-1])
                                object_start_end_ = str(o_idx_list_[0][0]) + "\t" + str(o_idx_list_[0][-1])
                                sent_seg_ = ''
                                for w in words_seg_updated:
                                    sent_seg_ += w + ' '
                                sent_seg_ = sent_seg_.strip()
                                output_train.write(str(line_id_train) + "\t" + "\"" + sentence + "\"" +
                                               "\t" + "\"" + sent_seg_ + "\"" + "\t" + sub + "\t" +
                                               subject_start_end_ + "\t" + obj + "\t" + object_start_end_ + "\n")
                                output_train.write(relation + "(subject,object)" + "\n\n")
                                line_id_train += 1
                                count_error_handled_train += 1
                            except:
                                print("ERROR")
                                with open(output_bad_case_path, "a+") as bad_case:
                                    bad_case.write("bad case-training set\n")
                                    log_f = sentence + "\n" + str(
                                        words_seg_updated) + "\n" + sub + "\t" + obj + "\t" + relation + "\n"
                                    bad_case.write(log_f)
                                    print(log_f)
                                bad_case.close()

            output_train.close()

        logger.info("process dev set....")
        with open(output_dev_path, "w") as output_dev:
            output_dev.write("id" + "\t" + "text" + "\t" + "text_seg" + "\t" +
                             "subject" + "\t" + "subject_start_end" + "\t" +
                             "object" + "\t" + "object_start_end" + "\n")  # text header
            with open(self.dev_path, "r") as dev_f:
                count_error_dev = 0
                count_error_handled_dev = 0
                line_id_dev = 0
                for li in tqdm(dev_f.readlines()):
                    # verify that the input format of each line meets the format
                    if not self._is_valid_input_data(li):
                        print('Format is error')
                        return None
                    dic = li.strip()
                    dic = json.loads(dic)
                    sentence = dic['text']
                    sentence = str(sentence).replace(" ", '').lower()
                    sentence_term_list = [str(item['word']).lower() for item in dic['postag']
                                          if item["word"] != " "]
                    token_idx_list = self._get_token_idx(sentence_term_list, sentence)
                    if not dic["spo_list"]:
                        continue
                    for spo in dic["spo_list"]:
                        sub = str(spo['subject']).replace(" ", '').lower()
                        obj = str(spo['object']).replace(" ", '').lower()
                        relation = self._p_map_eng_dict.get(str(spo["predicate"]).replace(" ", ''))
                        s_idx_list = self._cal_item_pos(self._add_item_offset(sub, sentence),
                                                        token_idx_list)  # [[0,1,2]]
                        o_idx_list = self._cal_item_pos(self._add_item_offset(obj, sentence),
                                                        token_idx_list)
                        try:
                            subject_start_end = str(s_idx_list[0][0]) + "\t" + str(s_idx_list[0][-1])
                            object_start_end = str(o_idx_list[0][0]) + "\t" + str(o_idx_list[0][-1])

                            sent_seg = ''
                            for w in sentence_term_list:
                                sent_seg += w + ' '
                            sent_seg = sent_seg.strip()
                            output_dev.write(str(line_id_dev) + "\t" + "\"" + sentence + "\"" +
                                             "\t" + "\"" + sent_seg + "\"" + "\t" + sub + "\t" +
                                             subject_start_end + "\t" + obj + "\t" + object_start_end + "\n")
                            output_dev.write(relation + "(subject,object)" + "\n\n")
                            line_id_dev += 1
                        except:  ##Optimize the participle according to the subject and object
                            count_error_dev += 1
                            sent_line_updated, words_seg_updated = self.deal_with_participle_less_cutting_problem(
                                sentence, sub, obj)
                            token_idx_list_ = self._get_token_idx(words_seg_updated, sent_line_updated)
                            s_idx_list_ = self._cal_item_pos(self._add_item_offset(sub, sent_line_updated),
                                                             token_idx_list_)
                            o_idx_list_ = self._cal_item_pos(self._add_item_offset(obj, sent_line_updated),
                                                             token_idx_list_)
                            try:
                                subject_start_end_ = str(s_idx_list_[0][0]) + "\t" + str(s_idx_list_[0][-1])
                                object_start_end_ = str(o_idx_list_[0][0]) + "\t" + str(o_idx_list_[0][-1])
                                sent_seg_ = ''
                                for w in words_seg_updated:
                                    sent_seg_ += w + ' '
                                sent_seg_ = sent_seg_.strip()
                                output_dev.write(str(line_id_dev) + "\t" + "\"" + sentence + "\"" +
                                                 "\t" + "\"" + sent_seg_ + "\"" + "\t" + sub + "\t" +
                                                 subject_start_end_ + "\t" + obj + "\t" + object_start_end_ + "\n")
                                output_dev.write(relation + "(subject,object)" + "\n\n")
                                line_id_dev += 1
                                count_error_handled_dev += 1
                            except:
                                print("ERROR")
                                with open(output_bad_case_path, "a+") as bad_case:
                                    bad_case.write("bad case-dev set\n")
                                    log_f = sentence + "\n" + str(
                                        words_seg_updated) + "\n" + sub + "\t" + obj + "\t" + relation + "\n"
                                    bad_case.write(log_f)
                                    print(log_f)
                                bad_case.close()

            with open(output_bad_case_path, "a+") as bad_case:

                log_f_train = "Wrong number of sentences in training set: " + str(count_error_train) + "\n" + \
                              "Number of sovled sentences in training set: " + str(count_error_handled_train) + \
                              "\n" + "Number of unresolved sentences in training set: " + str(
                              count_error_train - count_error_handled_train) + "\n\n"
                bad_case.write("\n\n")
                bad_case.write(log_f_train)
                print(log_f_train)

                log_f_dev = "Wrong number of sentences in dev set: " + str(count_error_dev) + "\n" + \
                            "Number of solved sentences in dev set: " + str(count_error_handled_dev) + "\n" + \
                            "Number of unresolved sentences in dev set: " + str(
                            count_error_dev - count_error_handled_dev)
                bad_case.write(log_f_dev)
                print(log_f_dev)
            bad_case.close()

        output_dev.close()

    def add_relationship_other_in_raw_data(self, train_output_path, dev_output_path):

        sub_obj_overlap_all_train, sub_obj_overlap_all_dev = self._valid_sub_obj_intersection()
        for item in sub_obj_overlap_all_dev:
            if item not in sub_obj_overlap_all_train:
                sub_obj_overlap_all_train.add(item)
        sub_obj_overlap_all_global = sub_obj_overlap_all_train
        logger.info("process training set")
        with open(train_output_path, mode="w", encoding='utf-8') as train_output_f:
            with open(self.train_path, "r") as train_f:
                for li in tqdm(train_f.readlines()):
                    # verify that the input format of each line meets the format
                    if not self._is_valid_input_data(li):
                        print('Format is error')
                        return None
                    dic = li.strip()
                    dic = json.loads(dic)
                    if "spo_list" not in dic:
                        continue
                    spo_list = list(dic["spo_list"])
                    if len(spo_list) > 1:
                        sub_temp = []
                        obj_temp = []
                        relation_temp = []
                        for spo in spo_list:
                            sub = str(spo['subject']).replace(" ", '').replace("《", '').replace("》", '')
                            sub_temp.append(sub)
                            obj = str(spo['object']).replace(" ", '').replace("《", '').replace("》", '')
                            obj_temp.append(obj)
                            relation = spo["predicate"]
                            relation_temp.append(relation)

                        sub_obj_tuple_list = list(zip(sub_temp, obj_temp))
                        sub_temp_set_list = list(set(sub_temp))
                        obj_temp_set_list = list(set(obj_temp))

                        # strategy 1 <<<Results responding to data2.0>>>
                        # for s in sub_temp_set_list:
                        #     for o in obj_temp_set_list:
                        #         temp = []
                        #         temp_reverse = []
                        #         temp.append(s)
                        #         temp.append(o)
                        #         temp_reverse.append(o)
                        #         temp_reverse.append(s)
                        #         temp_ = tuple(temp)
                        #         temp_reverse = tuple(temp_reverse)
                        #
                        #         if temp_ not in sub_obj_tuple_list and s != o:
                        #             sub_obj_tuple_list.append(temp_)
                        #         elif temp_reverse not in sub_obj_tuple_list and s != o:
                        #             sub_obj_tuple_list.append(temp_reverse)

                        # strategy 1.1 <<<Results responding to data2.0.1>>>

                        for s in sub_temp_set_list:
                            for o in obj_temp_set_list:
                                if s in sub_obj_overlap_all_global or o in sub_obj_overlap_all_global:
                                    temp = []
                                    temp_reverse = []
                                    temp.append(s)
                                    temp.append(o)
                                    temp_reverse.append(o)
                                    temp_reverse.append(s)
                                    temp_ = tuple(temp)
                                    temp_reverse = tuple(temp_reverse)

                                    if temp_ not in sub_obj_tuple_list and s != o:
                                        sub_obj_tuple_list.append(temp_)
                                    elif temp_reverse not in sub_obj_tuple_list and s != o:
                                        sub_obj_tuple_list.append(temp_reverse)
                                else:
                                    continue

                        # strategy 2
                        for s in sub_temp_set_list:
                            for s_ in sub_temp_set_list:
                                if s in sub_obj_overlap_all_global or s_ in sub_obj_overlap_all_global:
                                    temp = []
                                    temp.append(s)
                                    temp.append(s_)
                                    temp_ = tuple(temp)
                                    if temp_ not in sub_obj_tuple_list and s != s_:
                                        sub_obj_tuple_list.append(temp_)
                                else:
                                    continue
                        #
                        # # strategy 3
                        for o in obj_temp_set_list:
                            for o_ in obj_temp_set_list:
                                if o in sub_obj_overlap_all_global or o_ in sub_obj_overlap_all_global:
                                    temp = []
                                    temp.append(o)
                                    temp.append(o_)
                                    temp_ = tuple(temp)
                                    if temp_ not in sub_obj_tuple_list and o != o_:
                                        sub_obj_tuple_list.append(temp_)
                                else:
                                    continue

                        add_len = len(sub_obj_tuple_list) - len(relation_temp)
                        if add_len == 0:
                            train_output_f.write(json.dumps(dic, ensure_ascii=False))
                            train_output_f.write('\n')
                            continue
                        else:
                            for i in range(add_len):
                                relation_temp.append("其他")

                            sub_obj_tuple_list = tuple(sub_obj_tuple_list)
                            sub_unzip_list = list(list(zip(*sub_obj_tuple_list))[0])
                            obj_unzip_list = list(list(zip(*sub_obj_tuple_list))[1])
                            sub_add = sub_unzip_list[len(sub_temp):]
                            obj_add = obj_unzip_list[len(obj_temp):]
                            relation_add = relation_temp[len(sub_temp):]
                            try:
                                for i in range(len(sub_add)):
                                    spo_temp = {}
                                    spo_temp["predicate"] = relation_add[i]
                                    spo_temp["object_type"] = "unknown"
                                    spo_temp["subject_type"] = "unknown"
                                    spo_temp["object"] = obj_add[i]
                                    spo_temp["subject"] = sub_add[i]
                                    spo_list.append(spo_temp)
                                line_dict = {}
                                postag = dic["postag"]
                                line_dict["postag"] = postag
                                text = dic["text"]
                                line_dict["text"] = text
                                line_dict["spo_list"] = spo_list
                                train_output_f.write(json.dumps(line_dict, ensure_ascii=False))
                                train_output_f.write('\n')
                            except:
                                print("ERROR")
                    else:
                        train_output_f.write(json.dumps(dic, ensure_ascii=False))
                        train_output_f.write('\n')
                train_output_f.close()

        logger.info("process dev set")
        with open(dev_output_path, mode="w", encoding='utf-8') as dev_output_f:
            with open(self.dev_path, "r") as dev_f:
                for li in tqdm(dev_f.readlines()):
                    # verify that the input format of each line meets the format
                    if not self._is_valid_input_data(li):
                        print('Format is error')
                        return None
                    dic = li.strip()
                    dic = json.loads(dic)
                    if "spo_list" not in dic:
                        continue
                    spo_list = list(dic["spo_list"])
                    if len(spo_list) > 1:
                        sub_temp = []
                        obj_temp = []
                        relation_temp = []
                        for spo in spo_list:
                            sub = str(spo['subject']).replace(" ", '').replace("《", '').replace("》", '')
                            sub_temp.append(sub)
                            obj = str(spo['object']).replace(" ", '').replace("《", '').replace("》", '')
                            obj_temp.append(obj)
                            relation = spo["predicate"]
                            relation_temp.append(relation)

                        sub_obj_tuple_list = list(zip(sub_temp, obj_temp))
                        sub_temp_set_list = list(set(sub_temp))
                        obj_temp_set_list = list(set(obj_temp))

                        # # strategy 1 <<<Results responding to data2.0>>>
                        # for s in sub_temp_set_list:
                        #     for o in obj_temp_set_list:
                        #         temp = []
                        #         temp_reverse = []
                        #         temp.append(s)
                        #         temp.append(o)
                        #         temp_reverse.append(o)
                        #         temp_reverse.append(s)
                        #         temp_ = tuple(temp)
                        #         temp_reverse = tuple(temp_reverse)
                        #
                        #         if temp_ not in sub_obj_tuple_list and s != o:
                        #             sub_obj_tuple_list.append(temp_)
                        #         elif temp_reverse not in sub_obj_tuple_list and s != o:
                        #             sub_obj_tuple_list.append(temp_reverse)

                        # strategy 1.1 <<<Results responding to data2.0.1>>>
                        for s in sub_temp_set_list:
                            for o in obj_temp_set_list:
                                if s in sub_obj_overlap_all_global or o in sub_obj_overlap_all_global:
                                    temp = []
                                    temp_reverse = []
                                    temp.append(s)
                                    temp.append(o)
                                    temp_reverse.append(o)
                                    temp_reverse.append(s)
                                    temp_ = tuple(temp)
                                    temp_reverse = tuple(temp_reverse)

                                    if temp_ not in sub_obj_tuple_list and s != o:
                                        sub_obj_tuple_list.append(temp_)
                                    elif temp_reverse not in sub_obj_tuple_list and s != o:
                                        sub_obj_tuple_list.append(temp_reverse)
                                else:
                                    continue

                        # strategy 2
                        for s in sub_temp_set_list:
                            for s_ in sub_temp_set_list:
                                if s in sub_obj_overlap_all_global or s_ in sub_obj_overlap_all_global:
                                    temp = []
                                    temp.append(s)
                                    temp.append(s_)
                                    temp_ = tuple(temp)
                                    if temp_ not in sub_obj_tuple_list and s != s_:
                                        sub_obj_tuple_list.append(temp_)
                                else:
                                    continue

                        # strategy 3
                        for o in obj_temp_set_list:
                            for o_ in obj_temp_set_list:
                                if o in sub_obj_overlap_all_global or o_ in sub_obj_overlap_all_global:
                                    temp = []
                                    temp.append(o)
                                    temp.append(o_)
                                    temp_ = tuple(temp)
                                    if temp_ not in sub_obj_tuple_list and o != o_:
                                        sub_obj_tuple_list.append(temp_)
                                else:
                                    continue

                        add_len = len(sub_obj_tuple_list) - len(relation_temp)
                        if add_len == 0:
                            dev_output_f.write(json.dumps(dic, ensure_ascii=False))
                            dev_output_f.write('\n')
                            continue
                        else:
                            for i in range(add_len):
                                relation_temp.append("其他")

                            sub_obj_tuple_list = tuple(sub_obj_tuple_list)
                            sub_unzip_list = list(list(zip(*sub_obj_tuple_list))[0])
                            obj_unzip_list = list(list(zip(*sub_obj_tuple_list))[1])
                            sub_add = sub_unzip_list[len(sub_temp):]
                            obj_add = obj_unzip_list[len(obj_temp):]
                            relation_add = relation_temp[len(sub_temp):]
                            try:
                                for i in range(len(sub_add)):
                                    spo_temp = {}
                                    spo_temp["predicate"] = relation_add[i]
                                    spo_temp["object_type"] = "unknown"
                                    spo_temp["subject_type"] = "unknown"
                                    spo_temp["object"] = obj_add[i]
                                    spo_temp["subject"] = sub_add[i]
                                    spo_list.append(spo_temp)
                                line_dict = {}
                                postag = dic["postag"]
                                line_dict["postag"] = postag
                                text = dic["text"]
                                line_dict["text"] = text
                                line_dict["spo_list"] = spo_list
                                dev_output_f.write(json.dumps(line_dict, ensure_ascii=False))
                                dev_output_f.write('\n')
                            except:
                                print("ERROR")
                    else:
                        dev_output_f.write(json.dumps(dic, ensure_ascii=False))
                        dev_output_f.write('\n')
                dev_output_f.close()

    def create_jieba_seg_user_dict(self,):
        jieba_user_dict = []
        logger.info("create jieba userdict...")
        logger.info("process training set")
        with open(self.train_path, "r") as train_f:
            for li in tqdm(train_f):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(li):
                    print('Format is error')
                    return None
                dic = li.strip()
                dic = json.loads(dic)
                if "spo_list" not in dic:
                    continue
                for spo in dic["spo_list"]:
                    sub = spo["subject"]
                    obj = spo["object"]
                    jieba_user_dict.append(sub)
                    jieba_user_dict.append(obj)
            train_f.close()

        logger.info("process dev set")
        with open(self.dev_path, "r") as dev_f:
            for li in tqdm(dev_f):
                # verify that the input format of each line meets the format
                if not self._is_valid_input_data(li):
                    print('Format is error')
                    return None
                dic = li.strip()
                dic = json.loads(dic)
                if "spo_list" not in dic:
                    continue
                for spo in dic["spo_list"]:
                    sub = spo["subject"]
                    obj = spo["object"]
                    jieba_user_dict.append(sub)
                    jieba_user_dict.append(obj)
            dev_f.close()
        jieba_user_dict = set(jieba_user_dict)
        logger.info("Number of subject and object types in training set and dev set: " + str(len(jieba_user_dict)))
        with open("./data/jieba_userdict.txt", 'w') as output:
            for w in jieba_user_dict:
                output.write(w + "\n")
            output.close()

    def jieba_seg(self,):

        logger.info("processing training set")
        with open("reset_data.txt", "w") as results_output:
            count_seg_error = 0
            with open(self.train_path, "r") as train_f:
                for li in tqdm(train_f):
                    # verify that the input format of each line meets the format
                    if not self._is_valid_input_data(li):
                        print('Format is error')
                        return None
                    dic = li.strip()
                    dic = json.loads(dic)
                    if "spo_list" not in dic:
                        continue
                    jieba_userdict_temp = []
                    subject_line_temp = []
                    object_line_temp = []
                    sentence_term_list = [item["word"] for item in dic["postag"]]
                    text = dic["text"]
                    for spo in dic["spo_list"]:
                        sub = spo["subject"]
                        obj = spo["object"]
                        relation = spo["predicate"]
                        if sub in sentence_term_list and obj in sentence_term_list:
                            results_output.write("\"" + text + "\"" + "\t" + "\"" + sentence_term_list + "\"" + \
                                                 "\t" + relation + "\t" + sub + "\t" + obj + "\n")
                        else:
                            count_seg_error += 1
                print("count_seg_error: " + str(count_seg_error))


if __name__=="__main__":
    data_reader = DataReader(train_path=
                             './data/train_data.json',
                             dev_path=
                             './data/dev_data.json',
                             test_path='./data/test1_data_postag_multi_sub_obj.json')
    # all_rel, train_rel, dev_rel = data_reader.get_all_relationship()
    # print(all_rel, len(all_rel))
    # output:
    # {'出生地', '编剧', '成立日期', '主演', '作者', '制片人', '毕业院校', '朝代', '字', '出品公司', '身高', '首都', '修业年限', '官方语言', '所属专辑', '导演', '面积',
    # '父亲', '出生日期', '专业代码', '改编自', '妻子', '所在城市'', '气候', '目', '歌手', '简称', '注册资本', '作词', '母亲', '总部地点', '人口数量', '丈夫', '邮政编码
    #  ', '海拔', '嘉宾', '主角', '民族', '占地面积'} 49
    # data_reader.generate_relationship_quantity_distribution_diagram(zh2en_dict_path=
    #                                                                 "./data/target_labels/relationship_zh2en")
    # data_reader.get_sub_obj_bio(output_path="./sub_obj_extraction/NER-LSTM-CRF/data/") ##Convert the training set and the dev set to the bio format.
    # data_reader.statistical_sub_obi_ner_labels_category_nummbers(train_ner_path=
    #                                                              "./sub_obj_extraction/NER-LSTM-CRF/data/train.txt",
    #                                                              dev_ner_path=
    #                                                              "./sub_obj_extraction/NER-LSTM-CRF/data/dev.txt")
    # data_reader.analyse_text_length_in_raw_data()
    #data_reader.after_split_text_analyse_text_length_in_raw_data(train_output_path="./train_text.txt")
    # data_reader._valid_sub_obj_intersection() ##Count the overlapping number of subject and object.
    # data_reader.get_obj_bio()  ##Convert the training set and the dev set to the bio format(only object).
    # data_reader.get_character_level_vector_based_word2vector() ## Training character-level word vectors based on original text.
    # data_reader.convert_data_format_for_RE_BGRU_2ATT(output_train_path="./data/data_format_for_RE_BGRU_2ATT/train.txt",
    #                                                  output_dev_path="./data/data_format_for_RE_BGRU_2ATT/dev.txt")
    # data_reader.convert_data_format_for_entity_aware_relation_classification_1(p_eng_dict_path="./data/target_labels/relationship_zh2en",
    #                                                                          output_train_path='./entity_aware_dataset/entity_aware_train',
    #                                                                          output_dev_path='./entity_aware_dataset/entity_aware_dev')

    # data_reader.create_jieba_seg_user_dict() ##results are saved at "./data/"

    # data_reader.analyse_sub_obj_in_raw_data()
    data_reader.add_relationship_other_in_raw_data(train_output_path=
                                                   "./data/relationship_extraction_data/"
                                                   "add_relationship_other_in_raw_data2.0.4/train.json",
                                                   dev_output_path=
                                                   "./data/relationship_extraction_data/"
                                                   "add_relationship_other_in_raw_data2.0.4/dev.json")

    #add "other" relationship in the raw data and analyse it.
    data_reader2 = DataReader(train_path='./data/relationship_extraction_data/'
                                         'add_relationship_other_in_raw_data2.0.4/train.json',
                              dev_path='./data/relationship_extraction_data/'
                                       'add_relationship_other_in_raw_data2.0.4/dev.json',
                              test_path='./data/test1_data_postag_multi_sub_obj.json')
    data_reader2.generate_relationship_quantity_distribution_diagram(zh2en_dict_path=
                                                                     "./data/target_labels/relationship_zh2en")
    data_reader2.convert_data_format_for_entity_aware_relation_classification_2(
        p_eng_dict_path="./data/target_labels/relationship_zh2en",
        output_train_path='./relationship_extraction/entity-aware-relation-classification_50_relationship/data/update_20190425/'
                          'train',
        output_dev_path='./relationship_extraction/entity-aware-relation-classification_50_relationship/data/update_20190425/'
                        'dev',
        output_test_path='./relationship_extraction/entity-aware-relation-classification_50_relationship/data/update_20190425/'
                         'test',
        output_bad_case_path='./relationship_extraction/entity-aware-relation-classification_50_relationship/data/update_20190425/'
                             'bad_case_details.txt')






