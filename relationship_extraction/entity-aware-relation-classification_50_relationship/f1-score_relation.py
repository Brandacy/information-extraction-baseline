from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import utils
import numpy as np

dev_relation_truely = []
with open("./runs_logs/20190419_attach_sub_obj_markets/dev_target.txt", "r") as dev_target:
    for line in dev_target:
        dev_relation_truely.append(utils.class2label[line.split(" ")[1].replace("\n", "")])
dev_relation_truely_ = np.array(dev_relation_truely, dtype="int")

dev_relation_predictions = []
with open("./runs_logs/20190419_attach_sub_obj_markets/logs/predictions.txt", "r") as dev_predictions:
    for line in dev_predictions:
        dev_relation_predictions.append(utils.class2label[line.split("\t")[1].replace("\n", "")])
dev_relation_predictions_ = np.array(dev_relation_predictions, dtype="int")

precision = precision_score(dev_relation_truely_, dev_relation_predictions_, average='macro')
recall = recall_score(dev_relation_truely_, dev_relation_predictions_, average='macro')
f1 = f1_score(dev_relation_truely_, dev_relation_predictions_, average='macro')
labels = [v for v in utils.class2label.values()]
f = open("../../data/target_labels/relationship_zh2en", 'r')
target_names = [li.split("\t")[0] for li in f.readlines()]
classification_report_ = classification_report(dev_relation_truely_,
                                               dev_relation_predictions_, labels=labels,
                                               target_names=target_names)

P_R_F1_log = "<<< (50)-RELATIONSHIP EVALUATION RESULT ON DEV SET -- LIC2019 Information Extraction Task>>>:\n" \
                 "macro-averaged Precision = {:g}%\n".format(precision) + \
                 "macro-averaged Recall = {:g}%\n".format(recall) + \
                 "macro-averaged F1-score = {:g}%\n\n".format(f1) + \
                 classification_report_
with open("./runs_logs/20190419_attach_sub_obj_markets/logs/classification_report.txt", 'w') as report:
    report.write(P_R_F1_log)
report.close()
print(P_R_F1_log)

