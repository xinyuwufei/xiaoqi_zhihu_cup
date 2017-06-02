from itertools import chain, combinations
import json

def dump(d,file):

    with open(file,'w') as myfile:
        json.dump(d,myfile)

def load(file):
    with open(file,'r') as myfile:
        d=json.load(myfile)
    return d


def load_topic_info():

    """

    :param parent_to_child:
    :type parent_to_child:
    :return:
    :rtype:
    """

    d={}

    with open('../data/topic_info.txt', 'r') as f:
        # w.write('Source,Target\n')
        for line in f:
            line=line.strip().split('\t')
            child=line[0].strip()
            parents=line[1].strip().split(',')

            for p in parents:
                if not p:

                    if 'root' not in d:
                        d['root']=[child]
                    else:
                        d['root'].append(child)
                else:

                    if p not in d:
                        d[p] = [child]
                    else:
                        d[p].append(child)
        leafs=set()
        for key,value in d.items():
            for child in value:
                if child not in d:
                    leafs.add(child)
        print('# of leafs',len(leafs))

    return d,leafs

def load_question_topic_train_set():
    d = {}

    with open('../data/question_topic_train_set.txt', 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            question_id = line[0].strip()
            topic_id = line[1].strip().split(',')
            if question_id in d:
                print('question_id',question_id)
            else:
                d[question_id]=[x.strip() for x in topic_id]
    return d


def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def find_topic(topic_id,data,indexs=None):
    d=[]
    if indexs is None:
        for x in data:
            if topic_id in data[x]:
                d.append(x)
        return d
    elif not indexs:
        return d
    else:
        for i in indexs:
            if topic_id in data[i]:
                d.append(i)
        return d



def findsubsets(S, m):
    return set(combinations(S, m))





dict,leafs=load_topic_info()
question_topic=load_question_topic_train_set()

nodes={}
count=0

for ccc in leafs:
    print('ccc',ccc)
    paths = find_all_paths(dict, 'root', ccc)
    print('after finding path',len(paths))

    subtree_nodes=set()
    for path in paths:
        real_path=path[1:-1]
        for n in real_path:
            subtree_nodes.add(n)
    subtree_nodes.add(ccc)
    ncm=len(subtree_nodes)
    if ncm>5:
        ncm=5

    for i in range(1,ncm+1):
        s=findsubsets(subtree_nodes, i)
        for j,comb in enumerate(s):
            comb=sorted(comb)
            topic_id=comb[-1]
            child=','.join(comb)
            parent = ','.join(comb[0:-1])
            if child in nodes:
                continue
            elif i==1:
                nodes[child] = find_topic(topic_id, question_topic)
            else:
                nodes[child] = find_topic(topic_id, question_topic,
                                          nodes[parent])
            print('chd',child,'par',parent)
            print('\tlen of nodes[child]:',len(nodes[child]))
            print('after finding topic',j,i,len(subtree_nodes))
    count+=1
    print('#',count,'node:',ccc,'current_len:',len(nodes),'\n\n')


dump(nodes,'./path_joint_count_sorted1.json')



