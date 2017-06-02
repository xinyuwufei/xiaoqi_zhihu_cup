from itertools import chain, combinations
import tool

class Node(object):
    def __init__(self, data, prev, next):
        self.data = data
        self.prev = prev
        self.next = next


class Doubly_linked_list():
    head = None
    tail = None
    def Dfs_find(self,start_ptr,target):
        closed = []
        stack_fringe = []
        stack_fringe.append(start_ptr)

        while stack_fringe:
            current_ptr=stack_fringe.pop()
            if current_ptr.data==target:
                return current_ptr
            closed.append(current_ptr)
            if not current_ptr.next:
                continue
            for child_ptr in current_ptr.next:
                if child_ptr not in closed and child_ptr not in stack_fringe:
                    stack_fringe.append(child_ptr)
        return None

    def Dfs_find_ancestors(self,start_ptr):
        # a=set()
        # not_visited=[start_ptr.prev]
        # while not_visited:
        #     current=not_visited.pop()
        #     a.add()
        #     for x in current:
        #         # if x.data!='root':
        #         #     a.add(x.data)
        #         if x.prev:
        #             not_visited.append(x.prev)
        # return a

        #         closed = []
        # stack_fringe = []
        # stack_fringe.append(start_ptr)
        a=[]
        closed = []
        stack_fringe = []
        stack_fringe.append(start_ptr)
        while stack_fringe:
            current_ptr=stack_fringe.pop()
            if current_ptr.data!='root':
                a.append(current_ptr.data)
            closed.append(current_ptr)
            if not current_ptr.prev:
                continue
            for child_ptr in current_ptr.prev:
                if child_ptr not in closed and child_ptr not in stack_fringe:
                    stack_fringe.append(child_ptr)
        return list(set(a[1:]))

    def Dfs_find_descendants(self,start_ptr):
        # a=set()
        # not_visited=[start_ptr.next]
        # while not_visited:
        #     current=not_visited.pop()
        #     for x in current:
        #         a.add(x.data)
        #         if x.next:
        #             not_visited.append(x.next)
        # return a
        a=[]
        closed = []
        stack_fringe = []
        stack_fringe.append(start_ptr)
        while stack_fringe:
            current_ptr=stack_fringe.pop()
            a.append(current_ptr.data)
            closed.append(current_ptr)
            if not current_ptr.next:
                continue
            for child_ptr in current_ptr.next:
                if child_ptr not in closed and child_ptr not in stack_fringe:
                    stack_fringe.append(child_ptr)
        return list(set(a[1:]))

    def Dfs(self):
        d={}
        has_intersection={}
        closed = []
        stack_fringe = []
        stack_fringe.append(self.head)

        while stack_fringe:
            current_ptr = stack_fringe.pop()
            if current_ptr.data != 'root':
                if len(current_ptr.prev)>1:
                    has_intersection[current_ptr.data]=[x.data for x in current_ptr.prev]
                    print(current_ptr.data,len(current_ptr.prev))
                d[current_ptr.data]=[self.Dfs_find_ancestors(current_ptr)]
                d[current_ptr.data].append(
                    self.Dfs_find_descendants(current_ptr))
                # print(current_ptr.data,len(d[current_ptr.data][0]),len(d[current_ptr.data][1]))
            closed.append(current_ptr)
            if not current_ptr.next:
                continue
            for child_ptr in current_ptr.next:
                if child_ptr not in closed and child_ptr not in stack_fringe:
                    stack_fringe.append(child_ptr)
        tool.dump(d,'./summary/node_ancestors_descendants.json')
        tool.dump(has_intersection,'./summary/node_has_intersection.json')
    def append(self, parent,child):
        if self.head is None:
            child_node = Node(child, [], [])
            self.head = child_node
            return
        else:
            parent_node = self.Dfs_find(self.head, parent)
            # print('parent_node',parent_node,parent)
            # print('looking for child', child)
            child_node = self.Dfs_find(self.head, child)
            if child_node is None:
                child_node = Node(child,[],[])
            
            child_node.prev.append(parent_node)
            parent_node.next.append(child_node)



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







d,_=load_topic_info()
dll=Doubly_linked_list()
i=0
unvisited_queue=['root']
visited_path=set()
dll.append(None,'root')
while unvisited_queue:
    parent=unvisited_queue.pop(0)
    # closed.append(parent)
    # total.append(parent)
    if parent in d:
        for child in d[parent]:
            unvisited_queue.append(child)
            if parent+','+child not in visited_path:
                i+=1
                print(i,parent + ',' + child)
                dll.append(parent, child)
                visited_path.add(parent+','+child)
dll.Dfs()



# dll=Doubly_linked_list()
# dll.append(None,'A')
# dll.append('A','C')
# dll.append('A','B')
# dll.append('C','F')
# dll.append('B','F')
# dll.append('B','E')
# dll.append('E','F')
# dll.append('F','leaf')
# ptr = dll.Dfs_find(dll.head, 'F')
# print(len(ptr.prev))
# # for x in ptr.next:
# #     print(x.data)
# dll.Dfs_find_ancestors(ptr)
# # print(ptr.data)
# # print(dll.head.data,len(dll.head.next),dll.head.next[0].data)
# # print(dll.tail.data,len(dll.tail.prev))


