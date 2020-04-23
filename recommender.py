'''
@Author: wallfacer (Yanhan Zhang)
@Time: 2020/4/22 7:24 PM
'''

import pickle
from math import log2


class recommender:
    def __init__(self, top_k):
        self.top_k = top_k
        self.pred = pickle.load(open('pred_results.pkl', 'rb'))
        self.pred_x = self.pred[0]
        self.pred_y = self.pred[1]
        self.pred_result = self.pred[2]
        self.metrics = {}

    def parse_result(self):
        result = []
        count = 0
        for i in range(len(self.pred_y)):
            result.append(self.pred_result[count:count + len(self.pred_y[i])])
            count += len(self.pred_y[i])
        self.pred_result = result

    def recommend(self):
        self.recommends = []
        for i in range(len(self.pred_result)):
            cur_r = list(self.pred_result[i].reshape([-1]).argsort())
            cur_r.reverse()
            self.recommends.append(cur_r[:self.top_k])

    def compute_IDCG(self, n):
        result = 0.0
        for i in range(1, n+1):
            result += 1 / log2(1 + i)
        return result

    def compute_metrics(self):
        prec = []
        NDCG = []
        for i, y in enumerate(self.pred_y):
            tp = 0
            fp = 0
            DCG = 0.0
            for j, rec in enumerate(self.recommends[i]):
                if rec >= 4:
                    tp += 1
                    DCG += 1 / log2(2 + j)
                else:
                    fp += 1
            prec.append([tp, fp])
            IDCG = self.compute_IDCG(tp)
            if IDCG > 0.0:
                NDCG.append(DCG / IDCG)
        self.metrics['NDCG'] = sum(NDCG) / len(NDCG)
        micro = list(
            filter(lambda n: n > -0.5, list(map(lambda l: l[0] / (l[0] + l[1]) if (l[0] + l[1]) > 0 else -1, prec))))
        self.metrics['micro_precision'] = sum(micro) / len(micro)
        macro1 = sum(list(map(lambda l: l[0], prec)))
        macro2 = sum(list(map(lambda l: l[1], prec)))
        self.metrics['macro_precision'] = macro1 / (macro1 + macro2)

        recall = []
        for i, y in enumerate(self.pred_y):
            tp = 0
            fn = 0
            for j, label in enumerate(y):
                if label >= 4:
                    if j in self.recommends[i]:
                        tp += 1
                    else:
                        fn += 1
            recall.append([tp, fn])
        micro_r = list(
            filter(lambda n: n > -0.5, list(map(lambda l: l[0] / (l[0] + l[1]) if (l[0] + l[1]) > 0 else -1, recall))))
        self.metrics['micro_recall'] = sum(micro_r) / len(micro_r)
        macro1_r = sum(list(map(lambda l: l[0], recall)))
        macro2_r = sum(list(map(lambda l: l[1], recall)))
        self.metrics['macro_recall'] = macro1_r / (macro1_r + macro2_r)

        micro_f1 = []
        for i in range(len(prec)):
            if sum(prec[i]) == 0 or sum(recall[i]) == 0:
                pass
            else:
                micro_p = prec[i][0] / sum(prec[i])
                micro_r = recall[i][0] / sum(recall[i])
                if (micro_p + micro_r) > 0:
                    micro_f1.append(2 * micro_p * micro_r / (micro_p + micro_r))
        self.metrics['micro_f1'] = sum(micro_f1) / len(micro_f1)

        p = self.metrics['macro_precision']
        r = self.metrics['macro_recall']
        self.metrics['macro_f1'] = 2 * p * r / (p + r)
        for k, v in self.metrics.items():
            print(k, v)


if __name__ == '__main__':
    r = recommender(20)
    r.parse_result()
    r.recommend()
    r.compute_metrics()
    print('OK')
