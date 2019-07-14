import statistics as stats
import pickle

data_file ='./matrix_on_over.pickle'

def get_ap_rec(data):
    ap = []
    rec = []
    for key in data:
        (a,r) = data[key]
        ap.append(a)
        rec.append(r)
    return (ap,rec)


with open(data_file, 'rb') as f:
    data = pickle.load(f)

    max_aps = []
    max_recs = []
    for i in range(1,9):
        max_ap = 0
        max_rec = 0
        for j in range(4900, 5010, 10):
            (ap,rec) = data[str(i) + '_' + str(j)]
            max_ap = max(max_ap,ap)
            max_rec = max(max_rec,rec)
        max_aps.append(max_ap)
        max_recs.append(max_rec)
    print(stats.mean(max_aps))
    print(stats.stdev(max_aps))
    print(stats.mean(max_recs))
    print(stats.stdev(max_recs))
