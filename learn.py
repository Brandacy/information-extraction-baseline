import jieba
#text = "禅意 歌者 <object>  刘珂矣  </object> 《 <subject> 一袖云 </subject> 》 中 诉 知己 … 绵柔 纯净 的 女声 ， 将 心中 的 万水千山 尽意 勾勒 于 这 清素 画 音 中"
# split_flag = "love"
# text.insert(3, "2019")
# list=["love"]
# print(list[0], list[-1])
#text_split = text.split(split_flag)
#print(text_split)



# jieba.load_userdict("./data/jieba_userdict.txt")
# text_seg = "|".join(jieba.cut("苏州硕诺尔自动化设备有限公司图解文学常识：35天轻松学文学爱德华·尼科·埃尔南迪斯"))
# print(text_seg)
# text ='10700	"《 <subject> 撒旦校草吻过我 </subject> 》 是 一部 连载 于 17K小说网 的 小说 ， 作者 是 <object> 我 是 莫言 </object>"'
# text_eng = '8001	"The most common <e1>audits</e1> were about <e2>waste</e2> and recycling."'
# print(text.split("\t"))
# import time
# time = str(time.localtime(time.time()))
# print(time) ##output:time.struct_time(tm_year=2019, tm_mon=4, tm_mday=14, tm_hour=10, tm_min=52, tm_sec=2, tm_wday=6, tm_yday=104, tm_isdst=0)
# import re
# sentence = "《叶永烈科幻故事》是2006年新疆青少年出版的图书，作者是叶永烈"
# sub = "叶永烈科幻故事"
# obj = "叶永烈"
# s_pattern = re.compile(re.escape(obj), re.I)
#
#
# sub_list = re.findall(sub, sentence)
# obj_list = re.findall(obj, sentence)
# for m in s_pattern.finditer(sentence):
#     print(m.group(), m.start(), m.end())
# print(str(sub_list) + "\n" + str(obj_list) + "\n")
# text = " "
# text_ = text.split(" ")
# print(str(text_))

# from tqdm import tqdm
# for i in tqdm(range(10)):
#     print(i)
import pkuseg
#
seg = pkuseg.pkuseg(postag=True)           # 以默认配置加载模型
#text = seg.cut('《叶永烈科幻故事》是2006年新疆青少年出版的图书，作者是叶永烈')# 进行分词
# text2 = seg.cut('苏州硕诺尔自动化设备有限公司图解文学常识：35天轻松学文学爱德华·尼科·埃尔南迪斯')
text3 = seg.cut('《爱似苍穹》是juanchappamartíndeus执导，ken、chompooaraya、cherry、ploy主演的爱情剧')
# text4 = seg.cut('teichgräberniklas,德国籍足球运动员')
print(text3)
char_list = []
postag_list = []
for item in text3:
    word = item[0]
    pos_tag = str(item[1]).upper()
    if len(word) == 1:
        char_list.append(word)
        postag_list.append(str(pos_tag))
    else:
        char_temp = [c for c in word]
        postag_temp = [pos_tag]*len(char_temp)
        char_list.extend(char_temp)
        postag_list.extend(postag_temp)
c_p_zip = zip(char_list, postag_list)
for i in c_p_zip:
    print(i)



# print(text2)
# print(text3)
# print(text4)


# def _read_data(input_file):
#     """Reads a BIO data."""
#     with open(input_file) as f:
#         lines = []
#         words = []
#         labels = []
#         for line in f:
#             contends = line.strip()
#             word = line.strip().split(' ')[0]
#             label = line.strip().split(' ')[-1]
#             # if contends.startswith("-DOCSTART-"):
#             #     words.append('')
#             #     continue
#             # if len(contends) == 0 and words[-1] == '。':
#             if len(contends) == 0:
#                 l = ' '.join([label for label in labels if len(label) > 0])
#                 w = ' '.join([word for word in words if len(word) > 0])
#                 lines.append([l, w])
#                 words = []
#                 labels = []
#                 continue
#             words.append(word)
#             labels.append(label)
#         return lines
#
#
# lines = _read_data(input_file="./sub_obj_extraction/NER-LSTM-CRF/data/dev.txt")
# for line in lines[0:5]:
#     print(line)
