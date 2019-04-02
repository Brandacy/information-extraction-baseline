import json


def get_all_relationship(train_path='./data/train_data.json', dev_path='./data/dev_data.json'):
    with open(train_path, 'r', encoding='utf-8') as f:
        all_rel = []
        for line in f:
            try:
                dict = json.loads(line.strip())
                spo_list = dict["spo_list"]
                for dict_ in spo_list:
                    all_rel.append(dict_["predicate"])
            except:
                continue
        with open(dev_path, 'r', encoding='utf-8') as dev_f:
            for line in dev_f:
                try:
                    dict = json.loads(line.strip())
                    spo_list = dict["spo_list"]
                    for dict_ in spo_list:
                        if dict_["predicate"] not in all_rel:
                            all_rel.append(dict_["predicate"])
                except:
                    continue
    return set(all_rel)


if __name__=="__main__":
    all_rel = get_all_relationship(train_path='./data/train_data.json', dev_path='./data/dev_data.json')
    print(all_rel, len(all_rel))
    ##output:
    # {'出生地', '编剧', '成立日期', '主演', '作者', '制片人', '毕业院校', '朝代', '字', '出品公司', '身高', '首都', '修业年限', '官方语言', '所属专辑', '导演', '面积',
    #  '父亲', '出生日期', '专业代码', '改编自', '妻子', '所在城市'', '气候', '目', '歌手', '简称', '注册资本', '作词', '母亲', '总部地点', '人口数量', '丈夫', '邮政编码
    #  ', '海拔', '嘉宾', '主角', '民族', '占地面积'} 49

