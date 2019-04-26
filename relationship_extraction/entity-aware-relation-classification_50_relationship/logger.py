import subprocess
import os
import datetime
from configure import FLAGS
import utils
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class Logger:
    def __init__(self, out_dir):
        self.log_dir = os.path.abspath(os.path.join(out_dir, "logs"))
        os.makedirs(self.log_dir)
        self.log_path = os.path.abspath(os.path.join(self.log_dir, "logs.txt"))
        self.log_file = open(self.log_path, "w")

        self.print_hyperparameters()

        self.best_f1 = 0.0

    def print_hyperparameters(self):
        self.log_file.write("\n================ Hyper-parameters ================\n\n")
        for arg in vars(FLAGS):
            self.log_file.write("{}={}\n".format(arg.upper(), getattr(FLAGS, arg)))
        self.log_file.write("\n==================================================\n\n")

    def logging_train(self, step, loss, accuracy):
        time_str = datetime.datetime.now().isoformat()
        log = "{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)
        self.log_file.write(log+"\n")
        print(log)


    def logging_eval(self, step, loss, accuracy, predictions, dev_relation_truely):
        self.log_file.write("\nEvaluation:\n")
        # loss & acc
        time_str = datetime.datetime.now().isoformat()
        log = "{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)
        self.log_file.write(log + "\n")
        print(log)

        # f1-score
        prediction_path = os.path.abspath(os.path.join(self.log_dir, "predictions.txt"))
        prediction_file = open(prediction_path, 'w')
        for i in range(len(predictions)):
            prediction_file.write("{}\t{}\n".format(i, utils.label2class[predictions[i]]))
        prediction_file.close()

        precision = precision_score(dev_relation_truely, predictions, average='macro')
        recall = recall_score(dev_relation_truely, predictions, average='macro')
        f1 = f1_score(dev_relation_truely, predictions, average='macro')
        # perl_path = os.path.join(os.path.curdir,
        #                          "SemEval2010_task8_all_data",
        #                          "SemEval2010_task8_scorer-v1.2",
        #                          "semeval2010_task8_scorer-v1.2.pl")
        # target_path = os.path.join(os.path.curdir, "resource", "target.txt")
        # process = subprocess.Popen(["perl", perl_path, prediction_path, target_path], stdout=subprocess.PIPE)
        # str_parse = str(process.communicate()[0]).split("\\n")[-2]
        # idx = str_parse.find('%')
        # f1_score = float(str_parse[idx-5:idx])

        self.best_f1 = max(self.best_f1, f1)
        f1_log = "<<< (50)-RELATIONSHIP EVALUATION RESULT ON DEV SET -- LIC2019 Information Extraction Task >>>:\n" \
                 "macro-averaged Precision = {:g}%\n".format(precision) + \
                 "macro-averaged Recall = {:g}%\n".format(recall) + \
                 "macro-averaged F1-score = {:g}%, Best = {:g}%\n".format(f1, self.best_f1)
        self.log_file.write(f1_log + "\n")
        print(f1_log)
