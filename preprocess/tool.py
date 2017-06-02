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


def vocab_encoding():
	d={}
	with open('./vocab/topic_vocab_sorted.txt','r') as f:
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

def dump_csv(d,file):
	with open(file, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for key in sorted(d):
			writer.writerow([key]+d[key])
def load_csv(file):
	
	with open(file, 'r') as f:
		r=csv.reader(f)
		l = list(r)
	return l
