from sklearn.metrics import precision_score, f1_score, \
                            recall_score, accuracy_score

class ClassificationMetrics:
    def __init__(self):
        self.result_metrics = dict()

    def __calculate_metrics(self, y_true, y_pred, averages='macro'):
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        precision = precision_score(y_true= y_true, y_pred=y_pred, average=averages)
        recall = recall_score(y_true= y_true, y_pred=y_pred, average=averages)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average=averages)
        result_dict = {
            'precision':precision, 'accuracy': acc, 'recall': recall, 'f1_score': f1,
        }
        return result_dict

    def __call__(self, y_true, y_pred, averages='macro'):
        self.result_metrics = self.__calculate_metrics(y_true=y_true, y_pred=y_pred, averages=averages)
        return self.result_metrics