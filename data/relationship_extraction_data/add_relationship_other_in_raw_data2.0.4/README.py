
###output_results are used in ./relationship_extraction/entity-aware-relation-classification_50_relationship/data/update_20190425/

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
                                elif temp_reverse not in sub_obj_tuple_list and s != o: # ---->add
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
                                elif temp_reverse not in sub_obj_tuple_list and s != o: # ---->add
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