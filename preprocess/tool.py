import json,csv

def topic_vocab_to_topic_vocab_sorted():

	a=[]
	with open('./vocab/topic_vocab.txt','r') as f:
		for line in f:
			a.append(line.strip())
		a=sorted(a)

	with open('./vocab/topic_vocab_sorted.txt','w') as f:
		for line in a:
			f.write(line+'\n')

def question_id_vocab_sorted():
	a=[]

	with open('../data/question_topic_train_set.txt', 'r') as f:
		for line in f:
			line=line.strip().split('\t')
			a.append(line[0])
		a = sorted(a)
	with open('./vocab/question_id_vocab_sorted.txt', 'w')as f:
		for line in a:
			f.write(line + '\n')

def question_encoding():
	d={}
	with open('./vocab/question_id_vocab_sorted.txt', 'r')as f:
		for i, line in enumerate(f):
			d[line.strip()] = i
	return d

def question_topic_sorted():
	d={}
	question_enc=question_encoding()
	vocab_enc=vocab_encoding()
	with open('../data/question_topic_train_set.txt', 'r') as f:
		for line in f:
			line=line.strip().split('\t')
			topics=sorted([vocab_enc[x] for x in line[1].split(',') if x!=''])
			d[question_enc[line[0]]]=topics
	dump(d,'./summary/question_topic_sorted.json')


def vocab_encoding():
	d={}
	with open('./vocab/topic_vocab_topological_sorted.txt','r') as f:
	# with open('./vocab/topic_vocab_sorted.txt', 'r') as f:
		for i,line in enumerate(f):
			d[line.strip()]=i
	return d

def dump(d,file):
	with open(file,'w') as myfile:
		json.dump(d,myfile)
def load(file):
	with open(file,'r') as myfile:
		d=json.load(myfile)
	return d

def dump_csv(d,file,is_dict=True):
	with open(file, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
		if not is_dict:
			for x in d:
				writer.writerow(x)
		else:
			for key in sorted(d):
				writer.writerow([key]+d[key])

def load_csv(file,is_dict=False):
    l=[]
    with open(file, 'r') as f:
        r=csv.reader(f)
        l = list(r)
    if is_dict:
        d={}
        for ll in l:
            if ll[0] in d:
                d[ll[0]]+=ll[1:]
            else:
                d[ll[0]]=ll[1:]
        return d
    else:
        return l

# question_topic_sorted()




