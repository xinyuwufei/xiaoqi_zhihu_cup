import tool

def uv_graph_encode():
    d=[]
    l=tool.load_csv('./summary/uv_graph_no_incest.csv')
    for line in l:
        if line[0]!='root':
            d.append([vocab_enc[line[0]]]+[vocab_enc[line[1]]])
        else:
            d.append([line[0]]+[vocab_enc[line[1]]])
    tool.dump_csv(d,'./summary/uv_graph_no_incest_topological_sorted.csv',False)
def uv_graph_no_incest():
    l1=tool.load_csv('./summary/uv_graph.csv')
    l2=tool.load_csv('./summary/uv_graph_incest.csv')
    no_incest=[]
    for x in l1:
        if x not in l2:
            no_incest.append(x)
    tool.dump_csv(no_incest,'./summary/uv_graph_no_incest.csv',False)


def node_ancestors_descendants_count():
    d=tool.load('./summary/node_ancestors_descendants.json')
    dd={}
    for key in d:

        dd[vocab_enc[key]]=[len(d[key][0])]
        dd[vocab_enc[key]].append(len(d[key][1]))
    tool.dump_csv(dd,'./summary/node_ancestors_descendants_count.csv')


def node_question_belongs_to_ancestors_descendants(is_remove=False):

    joint = tool.load('./summary/path_joint_count_sorted1.json')
    ad=tool.load('./summary/node_ancestors_descendants.json')
    s={}
    for node,value in ad.items():
        ancestors=value[0]
        descendants=value[1]
        node_questions=set(joint[node])
        total=len(node_questions)
        # if not descendants:
        #     node_questions=total
        # else:


        for d in descendants:
            node_questions=node_questions-set(joint[d])
        not_belong_to_d=len(node_questions)
        node_questions=set(joint[node])
        for a in ancestors:
            node_questions=node_questions-set(joint[a])
        not_belong_to_a=len(node_questions)
        isa , isd='',''
        if not descendants:
            isd='no_descendants'
        if not ancestors:
            isa='no_ancestors'
        if is_remove:
            if not isa and not isd:
                s[vocab_enc[node]]=[total,not_belong_to_a,not_belong_to_d]
        else:
            s[vocab_enc[node]]=[total,not_belong_to_a,not_belong_to_d,isa,isd]
    if not is_remove:
        tool.dump_csv(s,'./summary/node_question_belongs_to_ancestors_descendants_count.csv')
    else:
        tool.dump_csv(s,'./summary/node_question_belongs_to_ancestors_descendants_count_is_remove.csv')

def node_associations_count(is_remove_0=False):
    d={}
    joint=tool.load('./summary/path_joint_count_sorted1.json')
    
    for node in joint:
        tmp=node.split(',')
        node_enc=','.join(str(vocab_enc[x]) for i,x in enumerate(tmp))
        if is_remove_0:
            length=len(joint[node])
            if length>0:
                d[node_enc]=[length]
        else:
            d[node_enc]=[len(joint[node])]
    if not is_remove_0:
        tool.dump_csv(d,'./summary/node_associations_count.csv')
    else:
        tool.dump_csv(d,'./summary/node_associations_count_is_remove_0.csv')

def node_has_intersection():
    d={}
    has_intersection=tool.load('./summary/node_has_intersection.json')
    for inter in has_intersection:
        d[vocab_enc[inter]]=[len(has_intersection[inter])]+[vocab_enc[x] for x in has_intersection[inter]]
    tool.dump_csv(d,'./summary/node_has_intersection.csv')

vocab_enc=tool.vocab_encoding()



# node_ancestors_descendants_count()
# node_question_belongs_to_ancestors_descendants(False)
# node_question_belongs_to_ancestors_descendants(True)
# node_associations_count(False)
# node_associations_count(True)
# node_has_intersection()


''''''
# uv_graph_no_incest()
uv_graph_encode()

