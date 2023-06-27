# A Node class for GBFS Pathfinding
class Node:
    def __init__(self, v, cost):
        self.v=v
        self.cost=cost

# pathNode class will help to store
# the path from src to dest.
class pathNode:
    def __init__(self, node, parent):
        self.node=node
        self.parent=parent

# Function to add edge in the graph.
def addEdge(u, v, cost):
    # Add edge u -> v with weight weight.
    adj[u].append(Node(v, cost))


# function to return key for any value
def get_key(val,my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"
 

# Best Fisrt Search algorithm
def BFS(src, dest, n, label):

    # Initializing priority queue and reached.
    PriorityQueue = []
    Reached = []
    g = [[] for i in range(n)]

    # Inserting src in priority queue.
    PriorityQueue.append(pathNode(get_key(src,label), None))
    g[0].append(0)

    # Iterating while the priority queue 
    # is not empty.
    while (PriorityQueue):

        currentNode = PriorityQueue[0]
        currentIndex = 0
        # Finding the node with the least value
        for i in range(len(PriorityQueue)):
            if(g[PriorityQueue[i].node][0] < g[currentNode.node][0]):
                currentNode = PriorityQueue[i]
                currentIndex = i

        # Removing the currentNode from 
        # the queue and adding it in 
        # the Reached.
        PriorityQueue.pop(currentIndex)
        Reached.append(currentNode)
        
        # If we have reached the destination node.
        if(currentNode.node == get_key(dest,label)):
            # Initializing the path and cost list. 
            path = []
            cur = currentNode
            cost=[]

            # Adding all the nodes in the 
            # path and cost list through which we have
            # reached to dest.
            while(cur != None):
                path.append(cur.node)
                cost.append(g[cur.node])
                cur = cur.parent
            

            # Reversing the path and cost list
            path.reverse()
            cost.reverse()
            print('----------------------BEST FIRST SEARCH---------------------')
            for i in range(len(path) - 1):
                for j in range(len(path) - 1):
                    if i==j:
                        print(label[path[i]],cost[i], end = " -> ")
            print(label[path[(len(path)-1)]],cost[(len(path)-1)])

        

        # Iterating over adjacents of 'currentNode'
        # and adding them to queue if 
        # they are neither in openList or closeList.
        for node in adj[currentNode.node]:
            Flag=True
            for i in range(len(PriorityQueue)):
                if(PriorityQueue[i].node == node.v):
                    if g[currentNode.node][0] + node.cost <  g[PriorityQueue[i].node][0]:
                        g[PriorityQueue[i].node][0] = g[currentNode.node][0] + node.cost
                        PriorityQueue.pop(i)
                        PriorityQueue.append(pathNode(node.v, currentNode))
                    Flag=False
            
            for x in Reached:
                if(x.node == node.v):
                    Flag=False
            
            if Flag==True:
                PriorityQueue.append(pathNode(node.v, currentNode))
                g[node.v].append(g[currentNode.node][0] + node.cost)

    return []
# A Star algorithm
def A_Star(h, src, dest, n, label):

    PriorityQueue = []
    Reached = []
    f = [[] for i in range(n)]
    g = [[] for i in range(n)]

    PriorityQueue.append(pathNode(get_key(src,label), None))
    f[0].append(h[0])
    g[0].append(0)

    
    while (PriorityQueue):
    
        currentNode = PriorityQueue[0]
        currentIndex = 0
        
        for i in range(len(PriorityQueue)):
            if(f[PriorityQueue[i].node][0] < f[currentNode.node][0]):
                currentNode = PriorityQueue[i]
                currentIndex = i

        
        PriorityQueue.pop(currentIndex)
        Reached.append(currentNode)
        
        
        if(currentNode.node == get_key(dest,label)):
            
            path = []
            cur = currentNode
            cost =[]

    
            while(cur != None):
                path.append(cur.node)
                cost.append(g[cur.node])
                cur = cur.parent
            

           
            path.reverse()
            cost.reverse()
            print('---------------------------A_STAR---------------------------')
            for i in range(len(path) - 1):
                for j in range(len(path) - 1):
                    if i==j:
                        print(label[path[i]],cost[i], end = " -> ")
            print(label[path[(len(path)-1)]],cost[(len(path)-1)])
        

        
        for node in adj[currentNode.node]:
            flag=True
            for i in range(len(PriorityQueue)):
                if(PriorityQueue[i].node == node.v):
                    if h[PriorityQueue[i].node] + g[currentNode.node][0] + node.cost <  f[PriorityQueue[i].node][0]:
                        f[PriorityQueue[i].node][0] = h[PriorityQueue[i].node] + g[currentNode.node][0] + node.cost
                        g[PriorityQueue[i].node][0] = g[currentNode.node][0] + node.cost
                        PriorityQueue.pop(i)
                        PriorityQueue.append(pathNode(node.v, currentNode))
                    flag=False
            
            for x in Reached:
                if(x.node == node.v):
                    flag=False
            
            if flag==True:
                PriorityQueue.append(pathNode(node.v, currentNode))
                f[node.v].append(g[currentNode.node][0] + node.cost + h[node.v])
                g[node.v].append(g[currentNode.node][0] + node.cost)

    return []
# Greedy Best First Search algorithm
def GBFS(h, src, dest, n, label):

    
    PriorityQueue = []
    Reached = []
    
    PriorityQueue.append(pathNode(get_key(src,label), None))

    
    
    while (PriorityQueue):

        currentNode = PriorityQueue[0]
        currentIndex = 0
        # Finding the node with the least 'h' value
        for i in range(len(PriorityQueue)):
            if(h[PriorityQueue[i].node] < h[currentNode.node]):
                currentNode = PriorityQueue[i]
                currentIndex = i

        
        PriorityQueue.pop(currentIndex)
        Reached.append(currentNode)
        
        # If we have reached the destination node.
        if(currentNode.node == get_key(dest,label)):
            
            path=[]
            cur = currentNode

           
            while(cur != None):
                path.append(cur.node)
                cur = cur.parent
            

           
            path.reverse()
            print('------------------GREEDY BEST FIRST SEARCH------------------')
            for i in range(len(path) - 1):
                for j in range(len(path) - 1):
                    if i==j:
                        print(label[path[i]], end = " -> ")
            print(label[path[(len(path)-1)]])
            
        
        for node in adj[currentNode.node]:
            flag=True
            for x in PriorityQueue:
                if(x.node == node.v):
                    flag=False
            
            for x in Reached:
                if(x.node == node.v):
                    flag=False
            
            if flag==True:
                PriorityQueue.append(pathNode(node.v, currentNode))

    return[]
# Read file function
def read_file(filename):
    with (open(filename,'r',encoding='utf-8')) as f:
        lst_ND=f.readlines()
        i=0
        label = dict()
        h=[]
        matrix=[]
    for dong in lst_ND:
        dong=dong.replace('\n','')
        if i==0:
            n=int(dong)
            i=i+1
            for j in range(n):
                adj.append([])
        elif (i in range(1,n+1)):
            label.update({i-1:dong})
            i=i+1
        elif (i in range(n+1,2*n+1)):
            h.append(dong)
            h = [int(x) for x in h]
            i=i+1
        else :
            if i != 2*n+1:
                dong=dong.split()
                dong.pop(0)
                u=[int(x) for x in dong]
                matrix.append(u)
            i=i+1
    for u in range(0,n):
        for v in range(0,n):
            if u!=v | matrix[u][v]!=-1:
                addEdge(u,v,matrix[u][v])
    return n,h,label



adj=[]
filename='romania.txt'
n,h,label=read_file(filename)

BFS('Arad','Bucharest',n,label)
GBFS(h, 'Arad', 'Bucharest', n, label)
A_Star(h, 'Arad', 'Bucharest',n,label)


