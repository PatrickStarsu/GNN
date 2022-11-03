import os
import networkx as nx
import numpy as np
import random
from six import iterkeys
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec, KeyedVectors

class Graph(defaultdict):
    """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""

    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def make_consistent(self):

        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):

        removed = 0

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        return self


class DeepWalk():

    def __init__(self,number_walks=10,walk_length=40):

        self.number_walks = number_walks
        self.walk_length = walk_length
        self.input = "D:/Jupyter/AI/GNN/GraphEmbedding/DeepWalk/example_graphs/cora.edgelist"
        self.input_root = 'D:/Jupyter/AI/GNN/GraphEmbedding/DeepWalk/data/cora'
        self.output = 'D:/Jupyter/AI/GNN/GraphEmbedding/DeepWalk/example_graphs/cora.embeddings'

    def load_edgelist(self, file_, undirected=True):

        G = Graph()  # Graph(<class 'list'>, {})
        with open(file_) as f:
            for l in f:
                x, y = l.strip().split()[:2]
                x = int(x)
                y = int(y)
                G[x].append(y)
                if undirected:
                    G[y].append(x)
        # G是一个字典  Graph(<class 'list'>, {35: [1033, 103482, 103515, 1050679,...],...}
        G.make_consistent()  # 对35的连边列表进行排序，在连边列表中去掉自己
        return G

    # 构建游走的序列
    def random_walk(self, G,path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.keys()))]   # 随机选择G的一个节点,e.g. path=[35]
            print(path)

        while len(path) < path_length:
            cur = path[-1]   # 当前节点
            if len(G[cur]) > 0:  # 当前节点有连边
                if rand.random() >= alpha:  # 随机游走到下一个节点
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]

    def build_deepwalk_corpus(self, G, num_paths, path_length, alpha=0, rand=random.Random(0)):
        walks = []

        nodes = list(G.nodes())

        for cnt in range(num_paths):  # 每个节点产生num_paths个游走序列，最后添加到walks里面
            rand.shuffle(nodes)
            for node in nodes:  # 每个节点都作为初始节点开始游走
                path = self.random_walk(G, path_length, rand=rand, alpha=alpha, start=node)
                if len(path) != self.walk_length:
                    print(len(path))
                walks.append(path)

        return walks

    def process(self):

        G = self.load_edgelist(self.input)  # Graph(<class 'list'>, {35: [887, 1033, 1688, 1956, ...], ...}
        print("Number of nodes: {}".format(len(G.nodes())))
        print("Walking...")
        walks = self.build_deepwalk_corpus(G, num_paths=self.number_walks,
                                            path_length=self.walk_length, alpha=0, rand=random.Random(0))
        print("Training...")
        model = Word2Vec(walks, size=64, window=5, min_count=0, sg=1, hs=1,
                         workers=1)
        model.wv.save_word2vec_format(self.output)

    def create_edgelist(self):
        # contains data from .content
        all_data = []
        # contains data from .cites
        all_edges = []

        for root, dirs, files in os.walk(self.input_root):
            for file in files:
                if '.content' in file:
                    with open(os.path.join(root, file), 'r') as f:
                        # print(f.read().splitlines())
                        all_data.extend(f.read().splitlines())
                        # print(all_data)
                elif 'cites' in file:
                    with open(os.path.join(root, file), 'r') as f:
                        all_edges.extend(f.read().splitlines())

        # parse the data from all_data and all_edge to create edgelist
        labels = []
        nodes = []
        X = []
        G = nx.Graph()

        for i, data in enumerate(all_data):
            elements = data.split('\t')
            labels.append(elements[-1])
            X.append(elements[1:-1])
            nodes.append(elements[0])

        X = np.array(X, dtype=int)
        N = X.shape[0]  # the number of nodes
        F = X.shape[1]  # the size of node features

        # parse the edge
        edge_list = []
        for edge in all_edges:
            e = edge.split('\t')
            G.add_edge(e[0], e[1])  # Building the edges to form a graph
            edge_list.append((e[0], e[1]))
        num_classes = len(set(labels))

        nx.write_edgelist(G, self.input, data=False)

        return nodes,labels

    def train_predict(self):

        nodes,labels = self.create_edgelist()
        DeepWalk(self.number_walks,self.walk_length).process()

        # 训练下游线性分类器得到预测结果
        model = KeyedVectors.load_word2vec_format(self.output, binary=False)
        # Numbering original labels
        LABEL = {
            'Case_Based': 1,
            'Genetic_Algorithms': 2,
            'Neural_Networks': 3,
            'Probabilistic_Methods': 4,
            'Reinforcement_Learning': 5,
            'Rule_Learning': 6,
            'Theory': 7
        }
        y = []

        # Original labels from .content file, map them to LABELS
        for temp in range(2708):
            y.append(LABEL[labels[temp]])
        y = np.array(y)

        x_train = []
        for i in range(2500):
            x_train.append(model[str(nodes[i])])
        x_train = np.array(x_train)
        x_test = []
        for i in range(2500, len(nodes)):
            x_test.append(model[str(nodes[i])])
        x_test = np.array(x_test)
        y_train = y[:2500]
        y_test = y[2500:]

        clf = LogisticRegression(solver="liblinear")
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        acc = (pred == y_test).sum() / len(pred)
        print(f"预测准确率为{acc * 100:.1f}%")

if __name__ == '__main__':

    DeepWalk().train_predict()

