from skmultiflow.data import FileStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection import PageHinkley, DDM
from skmultiflow.drift_detection.hddm_w import HDDM_W

from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.lazy import KNNClassifier
from skmultiflow.meta import AdditiveExpertEnsembleClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.rules import VeryFastDecisionRulesClassifier



def make_stream(classifier):
    stream = FileStream('./airlines.csv')
    ht = classifier
    evaluator = EvaluatePrequential(show_plot=True, pretrain_size=2000, max_samples=50000)
    evaluator.evaluate(stream=stream, model=ht)
    stream = evaluator.stream.y
    return stream


def eddm(stream):
    detected_change = []
    detected_warning = []
    eddm = EDDM()
    data_stream = stream
    for i in range(len(stream)):
        eddm.add_element(data_stream[i])
        if eddm.detected_warning_zone():
            detected_warning.append((data_stream[i]))
            print("Warning zone has been detected in data: {}"
                  " - of index: {}".format(data_stream[i], i))
        if eddm.detected_change():
            detected_change.append((data_stream[i]))
            print("Change has been detected in data: {}"
                  " - of index: {}".format(data_stream[i], i))
    print("EDDM Detected changes: " + str(len(detected_change)))
    print("EDDM Detected warning zones: " + str(len(detected_warning)))
    return str('Ilość wykrytych zmian dla algorytmu EDDM wynosi: ' + str(len(detected_change)))


def hddm_a(stream):
    detected_change = []
    detected_warning = []
    hddm_a = HDDM_A()
    data_stream = stream
    for i in range(len(stream)):
        hddm_a.add_element(data_stream[i])
        if hddm_a.detected_warning_zone():
            detected_warning.append((data_stream[i]))
            print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
        if hddm_a.detected_change():
            detected_change.append((data_stream[i]))
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    print("HDDM_A Detected changes: " + str(len(detected_change)))
    print("HDDM_A Detected warning zones: " + str(len(detected_warning)))
    return str('Ilość wykrytych zmian dla algorytmu HDDM_A wynosi: ' + str(len(detected_change)))


def ph(stream):
    detected_change = []
    ph = PageHinkley()
    data_stream = stream
    for i in range(len(stream)):
        ph.add_element(data_stream[i])
        if ph.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
            detected_change.append((data_stream[i]))
    print("PH Detected changes: " + str(len(detected_change)))
    return str('Ilość wykrytych zmian dla algorytmu PH wynosi: ' + str(len(detected_change)))


def hddm_w(stream):
    detected_change = []
    detected_warning = []
    hddm_w = HDDM_W()
    data_stream = stream
    for i in range(len(stream)):
        hddm_w.add_element(data_stream[i])
        if hddm_w.detected_warning_zone():
            print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
            detected_warning.append((data_stream[i]))
        if hddm_w.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
            detected_change.append((data_stream[i]))
    print("HDDM_W Detected changes: " + str(len(detected_change)))
    print("HDDM_W Detected warning zones: " + str(len(detected_warning)))
    return str('Ilość wykrytych zmian dla algorytmu HDDM_W wynosi: ' + str(len(detected_change)))


def ddm(stream):
    detected_change = []
    detected_warning = []
    ddm = DDM()
    data_stream = stream
    for i in range(len(stream)):
        ddm.add_element(data_stream[i])
        if ddm.detected_warning_zone():
            print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
            detected_warning.append((data_stream[i]))
        if ddm.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
            detected_change.append((data_stream[i]))
    print("DDM Detected changes: " + str(len(detected_change)))
    print("DDM Detected warning zones: " + str(len(detected_warning)))
    return str('Ilość wykrytych zmian dla algorytmu DDM wynosi: ' + str(len(detected_change)))


# Streams based on classifiers:
stream = make_stream(HoeffdingTreeClassifier())
# stream = make_stream(KNNClassifier(n_neighbors=8, max_window_size=2000, leaf_size=40))
# stream = make_stream(VeryFastDecisionRulesClassifier())
# stream = make_stream(AdditiveExpertEnsembleClassifier(n_estimators=5, base_estimator=NaiveBayes(nominal_attributes=None), beta=0.8, gamma=0.1, pruning='weakest'))
# stream = make_stream(AccuracyWeightedEnsembleClassifier(n_estimators=10, n_kept_estimators=30, base_estimator=NaiveBayes(nominal_attributes=None), window_size=200, n_splits=5))
# stream = make_stream(AdaptiveRandomForestClassifier(n_estimators=10)



#Drift detection:
ddm(stream)
# eddm(stream)
# hddm_a(stream)
# ph(stream)
# hddm_w(stream)
pass