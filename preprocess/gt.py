import networkx as nx
import matplotlib.pyplot as plt
import tool
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import mean
from itertools import combinations

def findsubsets(S, m):
    return set(combinations(S, m))

def jaccard_index(a,b):
    den=len(a|b)
    if den ==0:
        return 0.0
    else:
        return float(len(a&b) / den)

def siblings(G,a):
    sibs=set()
    for x in G.successors(a):
        for bb in G.predecessors(x):
            if bb !=a:
                sibs.add(bb)
    return sibs

def is_siblings(G,a,b):
    sibs=siblings(G,a)
    if b in sibs:
        return True
    else:
        return False

def min_max_mean_std_hist(data,title,x_describe,y_describe,color_id):
    plt.figure(figsize=(16, 8))
    # counts, bins, patches=plt.hist(data,bins=bin,color=color)
    # plt.xlabel(x_describe)
    # plt.ylabel(y_decribe)

    # bin_centers = 0.5 * np.diff(bins) + bins[:-1]



    # color = sns.color_palette()
    # cnt_srs = pd.Series(data).value_counts()
    # min=np.min(np.array(data))
    # max=np.max(np.array(data))
    # mean=np.mean(np.array(data))
    # std=np.std(np.array(data))
    # statx ='\nx: min: %.4f max: %.4f mean: %.4f std: %.4f\n' % (min, max, mean, std)
    min=np.min(np.array(cnt_srs.values))
    max=np.max(np.array(cnt_srs.values))
    mean=np.mean(np.array(cnt_srs.values))
    std=np.std(np.array(cnt_srs.values))

    staty='y: min: %.4f max: %.4f mean: %.4f std: %.4f\n' %(min,max,mean,std)
    plt.title(title+staty)
    sns.barplot(cnt_srs.index, cnt_srs.values, color=color[color_id])
    plt.ylabel(y_describe, fontsize=12)
    plt.xlabel(x_describe, fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()

    # for count, x,b in zip(counts, bin_centers,bins):
    #     # Label the raw counts
    #     plt.annotate(str(b), xy=(x, 0), xycoords=('data', 'axes fraction'),
    #                  xytext=(0, -35), textcoords='offset points', va='top',
    #                  ha='center')
    #     plt.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
    #         xytext=(0, -45), textcoords='offset points', va='top', ha='center')
    #
    #     # Label the percentages
    #     percent = '%0.0f%%' % (100 * float(count) / counts.sum())
    #     plt.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
    #         xytext=(0, -55), textcoords='offset points', va='top', ha='center')
    #
    #
    # # Give ourselves some more room at the bottom of the plot
    # plt.tight_layout(pad=4)

    # plt.show()




TG=nx.DiGraph()

edges=tool.load_csv('./summary/uv_graph_no_incest_topological_sorted.csv')
int_edges=[]
for e in edges:
    if e[0]=='root':
        int_edges.append([e[0],int(e[1])])
    else:
        int_edges.append([int(e[0]),int(e[1])])
TG.add_edges_from(int_edges)
TG.remove_node('root')




terminal_vertices=[x for x in TG.nodes_iter() if TG.out_degree(x)==0 and TG.in_degree(x)>0]
source_vertices=[x for x in TG.nodes_iter() if TG.in_degree(x)==0 and TG.out_degree(x)>0]
isolate_vertices=[x for x in TG.nodes_iter() if TG.in_degree(x)==0 and TG.out_degree(x)==0]

print(len(terminal_vertices),len(source_vertices),len(isolate_vertices))







# for x in isolate_vertices:
#     print(x)

# l=[]
# for node in nx.topological_sort(TG):
#     if node not in isolate_vertices:
#         l.append(node)
# print(len(l))
# for node in sorted(isolate_vertices):
#     l.append(node)
# print(len(l))
# with open('./vocab/topic_vocab_topological_sorted.txt','w')as f:
#     for line in l:
#         f.write(line+'\n')



# min_max_mean_std_hist([par[1] for par in par_chd],'# of parent distribution','# of parent','# of vertices',20)
# min_max_mean_std_hist([par[2] for par in par_chd],'# of child distribution','# of child','# of vertices',20,color='b')

# T_s=[(s,sum([1 for t in terminal_vertices if nx.has_path(TG,s,t)])) for s in source_vertices]
# S_t=[(t,sum([1 for s in source_vertices if nx.has_path(TG,s,t)])) for t in terminal_vertices]

#
# print(T_s)
# print(len(T_s))
# print(S_t)
# print(len(S_t))
#
#
#
#
# # P_s_t=[[s,t,sum([1 for _ in nx.all_simple_paths(TG,s,t)])] for t in terminal_vertices for s in source_vertices]
# # for x in P_s_t:
# #     print(x)
#
# D_s=[[s,len(nx.descendants(TG,s))] for s in source_vertices]
# A_t=[[t,len(nx.ancestors(TG,t))] for t in terminal_vertices]
# print(A_t[0])
#
# # any_pair_of_source=findsubsets(source_vertices,2)
# # D_s_dict={s:nx.descendants(TG,s) for s in source_vertices}
# #
# # J_D_s=[sorted(tuple)+[jaccard_index(D_s_dict[tuple[0]],D_s_dict[tuple[1]])] for tuple in any_pair_of_source]
# #
# # for x in J_D_s:
# #     print(x)
# #
# # any_pair_of_terminal=findsubsets(terminal_vertices,2)
# # A_t_dict={t:nx.ancestors(TG,t) for t in terminal_vertices}
# # J_A_t=[sorted(tuple)+[jaccard_index(A_t_dict[tuple[0]],A_t_dict[tuple[1]])] for tuple in any_pair_of_terminal]
# #
# # for x in J_A_t:
# #     print(x)
#
# S_t=[(t,[s for s in source_vertices if nx.has_path(TG,s,t)]) for t in terminal_vertices]
# print('im here')
# L_max_t=[]
# L_min_t=[]
# for x in S_t:
#     t=x[0]
#     sl=x[1]
#     tmp=[]
#     if sl:
#         for s in sl:
#             tmp+=[len(path) for path in nx.all_simple_paths(TG,s,t)]
#     if not tmp:
#         print('no path',x)
#         L_max_t.append([t,0])
#         L_min_t.append([t,0])
#     else:
#         L_max_t.append([t,max(tmp)])
#         L_min_t.append([t,min(tmp)])
#
#
# print(L_max_t)
# print(L_min_t)


'''about TAG-DOC Bipartite'''

TDB=tool.load('./summary/question_topic_sorted.json')


print('total number of questions on Training set:',len(TDB))

# question_cnt_over_num_labels=[]
# question_cnt_over_topic_ids=[]
# for t in TDB:
#     question_cnt_over_num_labels+=[len(TDB[t])]
#     for x in TDB[t]:
#         question_cnt_over_topic_ids.append(x)
# min_max_mean_std_hist(question_cnt_over_num_labels,'# of questions over # of labels','# of labels in questions','Question Occurrences',1)
# min_max_mean_std_hist(question_cnt_over_topic_ids,'# of questions over topics','topic_id','Question Occurrences',2)

# i,count=0,0
# ll=[]
# for q,topics in TDB.items():
#     if len(topics)>1:
#         print(topics)
#         subset=findsubsets(topics,2)
#         for comb in subset:
#             if nx.has_path(TG, str(comb[0]), str(comb[1])) or is_siblings(TG, str(comb[0]), str(comb[1])):
#                 count+=1
#                 break
#         # if count>0:ll.append(count)
#
#         i+=1
# print(count,i)

# j=0
# d={}
# dstat={}
# # TG.nodes_iter()
# for vertice in TG.nodes_iter():
#     Al=[]
#     Dl=[]
#     Sl=[]
#     A=nx.ancestors(TG,vertice)
#     D=nx.descendants(TG,vertice)
#     S=siblings(TG,vertice)
#     total=0
#     total_ads=0
#     if vertice in isolate_vertices:
#         print('isolated')

#     for q, topics in TDB.items():
#         if vertice in topics:
#             total+=len(topics)

#             for t in topics:
#                 if t in A:
#                     Al.append(t)
#                     total_ads+=1
#                 elif t in D:
#                     Dl.append(t)
#                     total_ads+=1
#                 elif t in S:
#                     Sl.append(t)
#                     total_ads+=1

#     d[vertice]=[Al,Dl,Sl,total_ads,total]
#     dstat[vertice]=[len(Al),len(Dl),len(Sl),total_ads,total]

#     print(vertice,'Al',len(Al))
#     print(vertice,'Dl',len(Dl))
#     print(vertice,'Sl',len(Sl))
#     print(total,total_ads)
#     # if Al:
#     #     print('al')
#     #     min_max_mean_std_hist(AL,
#     #                       'ancestors_vertice_'+str(vertice), 'ancestor_id',
#     #                       'Question Occurrences', 2)
#     # if Dl:
#     #     print('dl')
#     #     min_max_mean_std_hist(DL,
#     #                           'descendants_vertice_' + str(vertice),
#     #                           'descendant_id',
#     #                           'Question Occurrences', 2)
#     # if Sl:
#     #     print('sl')
#     #     min_max_mean_std_hist(AL,
#     #                           'siblings_vertice_' + str(vertice),
#     #                           'descendant_id',
#     # #                           'Question Occurrences', 2)
#     j+=1
#     print(j)

# tool.dump(d,'./summary/topic_dependency.json')
# tool.dump(dstat,'./summary/topic_dependency_stat.json')


tds=tool.load('./summary/topic_dependency_stat.json')
new_tds={'x':[],'y':[],'c':[]}
for k in sorted(tds):
    for i in range(3):
        new_tds['x'].append(k)
        new_tds['y'].append(tds[k][i]/tds[k][-1])
        if i==0:
            new_tds['c'].append('ancestors')
        elif i==1:
            new_tds['c'].append('descendants')
        else:
            new_tds['c'].append('siblings')

#     new_tds['ancestors'].append(tds[k][0])
#     new_tds['descendants'].append(tds[k][1])
#     new_tds['siblings'].append(tds[k][2])
#     new_tds['total'].append(tds[k][3])
# for i,x in enumerate(new_tds['ancestors']):
#     if x>1:
#         print(i)
#         break
# min_max_mean_std_hist_float(new_tds['ancestors'],'ancestors freq distribution','ancestors freq','# of vertices',20,'b')
plt.figure(figsize=(12, 8))
pp=sns.pointplot(x="x", y='y', hue="c", data=new_tds)
axes = pp.axes
axes.set_ylim(0,)
# sns.barplot(new_tds['x'], new_tds['y'])
plt.show()


