import json
from tqdm import tqdm


def load_p_eng_dict(dict_path):
    """load label dict from file"""
    p_eng2zh_dict_ = {}
    with open(dict_path, 'r') as fr:
        for li in fr:
            p_zh, p_eng = li.strip().split('\t')
            p_eng2zh_dict_[p_eng] = p_zh
    return p_eng2zh_dict_


p_eng2zh_dict = load_p_eng_dict("../data/target_labels/relationship_zh2en")

print("Generating the final submit result, please wait for a moment.")
with open("./data/20190425/test1_data_postag_single_sub_obj.json") as test_f1:
    dic_all_49p = {}
    for row_id, line in enumerate(test_f1.readlines(), 1):
        dic_temp = {}
        line = line.strip()
        dic = json.loads(line)
        dic_temp["postag"] = dic["postag"]
        dic_temp["text"] = dic["text"]
        dic_temp["spo_list"] = []
        dic_all_49p[row_id] = dic_temp
    test_f1.close()

    lines_49 = [line.strip() for line in open("./data/20190425/test1_49p")]
    lines_49_predict_labels = [line.strip().replace('\n', "")
                               for line in open("./data/20190425/test1_49_predict_labels.txt", 'r')]
    label_id = 0
    for idx in tqdm(range(0, len(lines_49), 3)):
        spo_dict = {}
        row_id = int(lines_49[idx].split("\t")[0])
        p = lines_49_predict_labels[label_id].split("(")[0]
        label_id += 1
        p_zh = p_eng2zh_dict[p]
        sub = lines_49[idx + 1].split("\t")[1]
        obj = lines_49[idx + 1].split("\t")[2]
        spo_dict["predicate"] = p_zh
        spo_dict["object_type"] = ''
        spo_dict["subject_type"] = ''
        spo_dict["object"] = obj
        spo_dict["subject"] = sub

        spo_list = dic_all_49p[row_id]["spo_list"]
        spo_list.append(spo_dict)
        dic_all_49p[row_id]["spo_list"] = spo_list

    with open("./results/result_part1.json", 'w') as result_49p:
        for dic in dic_all_49p.values():
            result_49p.write(json.dumps(dic, ensure_ascii=False))
            result_49p.write("\n")
    result_49p.close()


with open("./data/20190425/test1_data_postag_multi_sub_obj.json") as test_f2:
    dic_all_50p = {}
    for row_id, line in enumerate(test_f2.readlines(), 1):
        dic_temp = {}
        line = line.strip()
        dic = json.loads(line)
        dic_temp["postag"] = dic["postag"]
        dic_temp["text"] = dic["text"]
        dic_temp["spo_list"] = []
        dic_all_50p[row_id] = dic_temp
    test_f2.close()

    lines_50 = [line.strip() for line in open("./data/20190425/test1_50p")]
    lines_50_predict_labels = [line.strip().replace('\n', "")
                               for line in open("./data/20190425/test1_50_predict_labels.txt", 'r')]
    label_id = 0
    for idx in tqdm(range(0, len(lines_50), 3)):
        spo_dict = {}
        row_id = int(lines_50[idx].split("\t")[0])
        p = lines_50_predict_labels[label_id].split("(")[0]
        if p == 'OTHER':
            label_id += 1
            continue

        label_id += 1
        p_zh = p_eng2zh_dict[p]
        sub = lines_50[idx + 1].split("\t")[1]
        obj = lines_50[idx + 1].split("\t")[2]
        spo_dict["predicate"] = p_zh
        spo_dict["object_type"] = ''
        spo_dict["subject_type"] = ''
        spo_dict["object"] = obj
        spo_dict["subject"] = sub

        spo_list = dic_all_50p[row_id]["spo_list"]
        spo_list.append(spo_dict)
        dic_all_50p[row_id]["spo_list"] = spo_list

    with open("./results/result_part2.json", 'w') as result_50p:
        for dic in dic_all_50p.values():
            result_50p.write(json.dumps(dic, ensure_ascii=False))
            result_50p.write("\n")
    result_50p.close()







