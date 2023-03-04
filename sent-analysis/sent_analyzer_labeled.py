import os, sys
import numpy as np
import nlpnet
from random import shuffle as shuffle_list

from utils import vectorize
from model import get_model
from model import get_distances
from features import FeatureExtractor

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

from scipy import sparse

CLASSES = ['neg', 'neu', 'pos']
NEG, NEU, POS = 0, 1, 2 
THRESHOLD = 0

def help(*args):
	if args:
		print("-----------------------------------------------")
		print('error: ' + args[0])
		print('use -help for more information.')
		exit()

	print("-----------------------------------------------")
	print("Using sent_analyzer:\n")
	print("python3 sent_analyzer.py -mod <MOD> -classifier <CLASSIFIER> -params <FILE> -saveto <FOLDER> -verbose -evaluate <TYPE_OF_EVAL> <TRAIN_FILES> <TEST_FILES>")
	print()
	print("-mod <MOD> .... The type of running you desire.")
	print(" * 'CLASSIFY': for classification purposes.")
	print("      * 'saveto' option will write a file for statistics.")
	print(" * 'SELF':     for self-training corpus expansion")
	print(" * 'CO':       for co-training corpus expansion")
	print("\n-classifier <CLASSIFIER> .... which classifier is going to be used.")
	print(" * 'LinearSVM': SVM with a Linear kernel.")
	print(" * 'PolySVM': SVM with polynomial with degree of 3.")
	print(" * 'NB': Gaussian Naive Bayes classifier.")
	print(" * 'LR': Logistic Regression classifier.")
	print(" * 'MLP': Multi-layer Perceptron classifier.")
	print(" * 'ALL': It will use all of the above classifiers and present the results for comparison.")
	print("\n-params <FILE> .... path to a file with parameters for data representation, type '--help-params' for further instructions.")
	print("\n-saveto <FOLDER> .... path to the folder where the data will be stored. Please, use a not existant folder. Don't be a douche about it.")
	print("\n-verbose .... it will basically print everything.")
	print("\n-evaluate <TYPE_OF_EVAL> <TRAIN_FILES> <TEST_FILES> .... type of evaluation followed by files to be used in it.")
	print(" * Type of evaluation can be:")
	print("   * 10-fold : for 10-fold cross validation. No need of test files for this. (5 and 3 folds are also available)")
	print("   * test : for train/test evaluation.")
	print(" * <TRAIN_FILES> must be named with extensions '.pos', '.neu', '.neg'. The path must to the folder with the preffix.")
	print(" * <TEST_FILES> follows the same as above. Again, don't be a douche... that's how it is.")
	print("\n Thanks for using. Hope you can be satisfied to the results you may obtain.")
	exit()

''' load_data <file_name, shuffle=True>
	Receives the path to a file, reads it and returns a list with three lists.
	Input:
	 - file_name <str> : path to file.
	 - shuffle <bool> : shuffle or not the data.
	Output:
	 - list[sentences, labels, ids]
	   - sentences: the sentences of the data.
	   - labels   : the labels (0,1,2).
	   - ids      : the ids for each tweet.
'''
def load_data(file_name, shuffle=True, verbose=False):
	global CLASSES
	file_sentences = []
	file_labels = []
	file_ids = []
	for label, c in enumerate(CLASSES):
		if verbose: print('reading ' + file_name)
		with open('%s.%s' % (file_name, c), 'r', encoding='utf8') as f:
			for line in f:
				if len(' '.join(line.strip().split()[1:]).strip()) < 1:
					print('-out ['+line.strip().split()[0]+']')
					continue
				file_sentences.append(' '.join(line.strip().split()[1:]))
				file_ids.append(int(line.strip().split()[0]))
				#file_sentences.append(data[1:])
				file_labels.append(label)
				#file_ids.append(int(data[0]))
	file_data = [file_sentences, file_labels, file_ids]
	if shuffle:
		file_indexes = list(range(len(file_sentences)))
		shuffle_data = lambda v, ind: [v[i] for i in ind]
		shuffle_list(file_indexes)
		for i in range(len(file_data)):
			file_data[i] = shuffle_data(file_data[i], file_indexes)
	return file_data


def run_self(classifier='linearsvm', options='bow negation emoticon emoji senti_words postag verbose', percent='0.1', fs=False,
			 verbose=False, train_file='', embedding_file=False, nb_classes=3):
	global THRESHOLD
	
	# Changing number of classes from 3 to 2
	if nb_classes == 2:
		global CLASSES
		global POS 
		
		CLASSES = ['neg', 'pos']
		POS = 1
	
	#Setting flags
	f_bow = False
	f_negation = False
	f_emoticon = False
	f_emoji = False
	f_senti_words = False
	f_postag = False

	#Initiating threshold
	adding_threshold = 0

	for op in options:
		if 'bow' in options:
			f_bow = True
		if 'negation' in options:
			f_negation = True
		if 'emoticon' in options:
			f_emoticon = True
		if 'emoji' in options:
			f_emoji = True
		if 'senti_words' in options:
			f_senti_words = True
		if 'postag' in options:
			f_postag = True

	percent = float(percent)

	#LOAD TRAIN FILES
	if verbose: print("################################################################")
	if verbose: print('loading data...')
	cp_sentences, cp_labels, cp_ids = load_data(train_file,shuffle=True)
	
	#################################################
	
	split_num = int(1/percent)
	
	skf = StratifiedKFold(n_splits=10)
	placeholder = np.zeros(len(cp_sentences))
	unlabeled_idx = None
	labeled_idx = None

	for unlabeled_idx, labeled_idx in skf.split(placeholder,cp_labels):
		break
	
	unlabeled_sentences = []
	unlabeled_labels = []
	unlabeled_ids = []

	data_sentences = []
	data_labels = []
	data_ids = []

	for i in unlabeled_idx:
		unlabeled_sentences.append(cp_sentences[i])
		unlabeled_labels.append(cp_labels[i])
		unlabeled_ids.append(cp_ids[i])
	for i in labeled_idx:
		data_sentences.append(cp_sentences[i])
		data_labels.append(cp_labels[i])
		data_ids.append(cp_ids[i])

	corpus_sentences = []
	corpus_ids = []
	corpus_labels = []
	corpus_gold = []

	THRESHOLD = int(len(unlabeled_ids)*percent)

	if verbose: print('Threshold: ' + str(THRESHOLD) + '/' + str(percent))

	if embedding_file != False:
		if verbose: print('reading embedding file')
		'''embeddings'''
		feats = FeatureExtractor(emb='Word2Vec', embedding_file=embedding_file, verbose=False)
	else:

		if verbose: print('initializing features')
		feats = FeatureExtractor(bow=f_bow, negation=f_negation, emoticon=f_emoticon, emoji=f_emoji, senti_words=f_senti_words,
			                     postag=f_postag, verbose=False)
		feats.make_bow(data_sentences, unlabeled_sentences)

	if verbose: print('classifier: ' + classifier + '\n')

	if verbose: print('extracting labeled document features')
	X_train = feats.get_representation(data_sentences)
	Y_train = np.array(data_labels)

	if verbose: print('extracting unlabeled document features')
	X_extend  = feats.get_representation(unlabeled_sentences)

	#Feature selection
	if fs and embedding_file == False:
		if verbose: print(X_train.shape)
		fs_clf = LinearSVC(C=0.25, penalty="l1", dual=False, random_state=1).fit(X_train,Y_train)
		X_train = SelectFromModel(fs_clf,prefit=True).transform(X_train)
		X_extend = SelectFromModel(fs_clf,prefit=True).transform(X_extend)
		if verbose: print(X_train.shape)

	for i in range(1,int(1/percent)+1):
			
		if verbose: print('iteration: %i' % i)
		model = get_model(model_name=classifier)
			
		if verbose: print('\n training classifier')
		model.fit(X_train, Y_train)

		if verbose: print(' predicting labels')
		# predição
		predictions = model.predict(X_extend)
			
		if verbose: print('\nmeasuring reliability')
		# confiabilidade
		distances = get_distances(model, X_extend, model_name=classifier)

		if verbose:
			print('Current acc: %f' % np.mean(predictions == unlabeled_labels))
			f1 = f1_score(unlabeled_labels, predictions, average=None)
			print("Current F1.",end='')
			for cl in range(0,len(CLASSES)): print(' %s: %.3f' % (CLASSES[cl].lower(), f1[cl]), end='')
			print(' -> %f' % np.mean(f1))
			print(confusion_matrix(unlabeled_labels, predictions))

		reliable_predictions = []
		for sent_id, (sent_dist, sent_pred) in enumerate(zip(distances, predictions)):
			dist_to_predicted_label = sent_dist[sent_pred]
			reliable_predictions.append((sent_id, dist_to_predicted_label, sent_pred))
			
		adding_idx, adding_dists, adding_preds = zip(*filter_continous(reliable_predictions))
		adding_idx = np.array(adding_idx)
		keep_idx, keep_dists, keep_preds = zip(*filter_continous_opposite(reliable_predictions))
		keep_idx = np.array(keep_idx)

		if verbose:
			acertos_label = len(CLASSES) * [0]
			total_label = len(CLASSES) * [0]
			total_acertos = 0
			for add_i, add in enumerate(adding_idx):
				current_label = adding_preds[add_i]
				gold_label = cp_labels[cp_ids.index(unlabeled_ids[add])]
				if current_label == gold_label:
					acertos_label[gold_label] += 1
					total_acertos += 1
				total_label[gold_label] += 1
			print('\nAcertos p/doc: %d/%d (%f)' % (total_acertos,len(adding_idx),total_acertos/len(adding_idx)))
			print('Acertos p/classe: ',end='')
			for anyVar in range(0,len(CLASSES)): print(str(acertos_label[anyVar]) + '/' +str(total_label[anyVar]),end=' ')
			print('')

			
		adding_preds = predictions[adding_idx]

		print("Adding %d - " % len(adding_preds),end='')	
		for c in range(0,len(CLASSES)): print(np.count_nonzero(adding_preds == c),end=' ')
		print(" | %d/%d" % (len(adding_idx),len(corpus_ids)))

		X_train = sparse.vstack([X_train, X_extend[adding_idx]])
		#X_train = np.concatenate([X_train, X_extend[adding_idx]])
		Y_train = np.concatenate([Y_train, adding_preds])
		X_extend = X_extend[keep_idx]



		#Adding data to final corpora
		for count, j in enumerate(adding_idx):
			corpus_ids.append(unlabeled_ids[j])
			corpus_labels.append(adding_preds[count])
			corpus_sentences.append(unlabeled_sentences[j])
			
		new_ui = []
		new_us = []
		new_ul = []
		for idx in keep_idx:
			new_ui.append(unlabeled_ids[idx])
			new_us.append(unlabeled_sentences[idx])
			new_ul.append(unlabeled_labels[idx])
		unlabeled_sentences = new_us
		unlabeled_ids = new_ui
		unlabeled_labels = new_ul

	if verbose: print('iteration: %i' % int((1/percent)+1))
	model = get_model(model_name=classifier)
			
	if verbose: print('\n training classifier')
	model.fit(X_train, Y_train)

	if verbose: print(' predicting labels')
	# predição
	predictions = model.predict(X_extend)
			
	if verbose: print('\nmeasuring reliability')
		
	for count, j in enumerate(predictions):
		corpus_ids.append(unlabeled_ids[count])
		corpus_labels.append(j)
		corpus_sentences.append(unlabeled_sentences[count])

	print("Adding %d - " % len(predictions),end='')	
	for c in range(0,len(CLASSES)): print(np.count_nonzero(predictions == c),end=' ')
	print(" | 0/%d" % len(corpus_ids))

	if verbose:
		print('Current acc: %f' % np.mean(predictions == unlabeled_labels))
		f1 = f1_score(unlabeled_labels, predictions, average=None)
		print("Current F1.",end='')
		for cl in range(0,len(CLASSES)): print(' %s: %.3f' % (CLASSES[cl].lower(), f1[cl]), end='')
		print(' -> %f' % np.mean(f1))
		print(confusion_matrix(unlabeled_labels, predictions))

	if verbose:
		acertos_label = len(CLASSES) * [0]
		total_label = len(CLASSES) * [0]
		total_acertos = 0
		for add_i, add in enumerate(unlabeled_ids):
			current_label = predictions[add_i]
			gold_label = cp_labels[cp_ids.index(unlabeled_ids[add_i])]
			if current_label == gold_label:
				acertos_label[gold_label] += 1
				total_acertos += 1
			total_label[gold_label] += 1
		print('\nAcertos p/doc: %d/%d (%f)' % (total_acertos,len(unlabeled_ids),total_acertos/len(unlabeled_ids)))
		print('Acertos p/classe: ',end='')
		for anyVar in range(0,len(CLASSES)): print(str(acertos_label[anyVar]) + '/' +str(total_label[anyVar]),end=' ')
		print('')

	gold_labels = []
	gold_ids = []
	for i in corpus_ids:
		gold_labels.append(cp_labels[cp_ids.index(i)])
		gold_ids.append(cp_ids[cp_ids.index(i)])

	if verbose:
		print('Final ----------------------------')
		print('Acc: %f' % np.mean(np.array(corpus_labels) == np.array(gold_labels)))
		f1 = f1_score(gold_labels, corpus_labels, average=None)
		print("F1.",end='')
		for cl in range(0,len(CLASSES)): print(' %s: %.3f' % (CLASSES[cl].lower(), f1[cl]), end='')
		print(' -> %f' % np.mean(f1))
		print(confusion_matrix(gold_labels, corpus_labels))


def report(distances, predictions, Y_test):
	acc_pred = np.mean(predictions == Y_test)
	f1 = f1_score(Y_test, predictions, average=None)
	print('Acc: %.4f' % accuracy_score(Y_test, predictions))
	#print('Acc: %.4f' % acc_pred)
	print("F1. neg: %.3f neu: %.3f post: %.3f" % (f1[0], f1[1],f1[2]))
	
	acc_dist_min = np.mean(np.argmin(distances, axis=-1) == Y_test)
	acc_dist_max = np.mean(np.argmax(distances, axis=-1) == Y_test)
	print('Acc dist min: %.4f' % acc_dist_min)
	print('Acc dist max: %.4f' % acc_dist_max)
	
	acc_dist_min_equal = np.mean((np.argmin(distances, axis=-1) == Y_test) & (predictions != Y_test))
	acc_dist_max_equal = np.mean((np.argmax(distances, axis=-1) == Y_test) & (predictions == Y_test))
	print('Acc dist min equal: %.4f' % acc_dist_min_equal)
	print('Acc dist max equal: %.4f' % acc_dist_max_equal)

def averageProb(reliable_predictions):
	#print(reliable_predictions[-1])
	values = []
	pos = []
	neu = []
	neg = []
	for i in reliable_predictions:
		values.append(i[1])
		if i[2] == 0:
			neg.append(i[1])
		if i[2] == 1:
			neu.append(i[1])
		if i[2] == 2:
			pos.append(i[1])
	print('POS avg: %f | dp: %f | var: %f' % (np.average(pos),np.std(pos),np.var(pos)))
	print('NEU avg: %f | dp: %f | var: %f' % (np.average(neu),np.std(neu),np.var(neu)))
	print('NEG avg: %f | dp: %f | var: %f' % (np.average(neg),np.std(neg),np.var(neg)))





def sort_distances(reliable_predictions):
	return sorted(reliable_predictions, key=lambda x: x[1], reverse=True)

def filter_continous(reliable_predictions):
	new_rp = sort_distances(reliable_predictions)
	return new_rp[:THRESHOLD]

def filter_continous_opposite(reliable_predictions):
	new_rp = sort_distances(reliable_predictions)
	return new_rp[THRESHOLD:]

def filter_continous2(reliable_predictions, adding_threshold):
	new_rp = sort_distances(reliable_predictions)
	if new_rp[0][1] < adding_threshold:
		print(new_rp[0][1],end=' ')
		print('on')
		return None

	for i in range(0,len(new_rp)):
		if new_rp[i][1] < adding_threshold:
			return new_rp[:i]
	return new_rp
	
def filter_continous2_opposite(reliable_predictions, adding_threshold):
	new_rp = sort_distances(reliable_predictions)
	if new_rp[-1][1] >= adding_threshold:
		print(new_rp[-1][1],end=' ')
		print('over')
		return None
	for i in range(0,len(new_rp)):
		if new_rp[i][1] < adding_threshold:
			return new_rp[i:]

def filter_balanced(reliable_predictions, adding_threshold):
	reliable_predictions = sort_distances(reliable_predictions)
	new_rp = []
	temp_rpc = []
	for c in range(len(CLASSES)):
		new_rp_c = list(filter(lambda x: x[2] == c, reliable_predictions))
		new_rp_c = list(filter(lambda x: x[1] >= adding_threshold, new_rp_c))
		temp_rpc.append(new_rp_c)
	
		new_rp_c = new_rp_c[:int(THRESHOLD/len(CLASSES))]
		new_rp.extend(new_rp_c)
	return new_rp

def filter_balanced_opposite(reliable_predictions, adding_threshold):
	reliable_predictions = sort_distances(reliable_predictions)
	new_rp = []
	for c in range(len(CLASSES)):
		new_rp_c = list(filter(lambda x: x[2] == c, reliable_predictions))
		new_rp_c = list(filter(lambda x: x[1] >= adding_threshold, new_rp_c))
		tmp = int(THRESHOLD/len(CLASSES))
		if int(THRESHOLD/len(CLASSES)) > len(new_rp_c):
			tmp = 0
		new_rp_c = new_rp_c[tmp:]
		new_rp.extend(new_rp_c)
	return new_rp
	
if __name__ == '__main__':
	
	#default options
	

	classifier='linearsvm'
	options='bow negation emoticon emoji senti_words postag verbose'
	percent='0.1'
	emb = False
	train_file='data/corpus/trainTT'
	fs = True
	verbose = False
	md = 'self'
	nb_classes = 2

	i = 0
	while i <= len(sys.argv[1:]):
		if sys.argv[i].lower() == '-help':
			help()
		if sys.argv[i].lower() == '-fs':
			fs = True
		if sys.argv[i].lower() == '-v' or sys.argv[i].lower() == '-verbose':
			verbose = True
		if sys.argv[i].lower() == '-classifier':
			classifier = sys.argv[i+1]
			if classifier not in ['linearsvm', 'polysvm','nb','lr','mlp','trees','randfor']:
				help('not a valid classifier')
			i += 1
		if sys.argv[i].lower() == '-options':
			options = sys.argv[i+1]
			i += 1
			options = options.replace(',',' ')
		if sys.argv[i].lower() == '-mode':
			md = sys.argv[i+1].lower()
			if md not in ['self', 'co']:
				help('not a valid mode')
			i += 1
		if sys.argv[i].lower() == '-percent':
			percent = sys.argv[i+1]
			i += 1
		if sys.argv[i].lower() == '-nb_classes':
			nb_classes = int(sys.argv[i+1])
			i += 1
		if sys.argv[i].lower() == '-train_file':
			train_file = sys.argv[i+1]
			i += 1
		if sys.argv[i].lower() == '-emb':
			emb = sys.argv[i+1]
			i += 1
		i += 1

	if md == 'self':
		run_self(classifier=classifier, options=options, percent=percent, verbose=verbose, fs=fs, train_file=train_file, 
			embedding_file=emb, nb_classes=nb_classes)
