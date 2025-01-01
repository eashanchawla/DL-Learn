from .base import Value
from graphviz import Digraph

def traverse(current: Value, nodes=None, edges=None):
    '''traverse to build nodes and edges for computational graph'''
    if nodes is None:
        nodes = []
    if edges is None:
        edges = []
    if current not in nodes:
        nodes.append(current)
        for parent in current._parents:
            edges.append((parent, current))
            traverse(parent, nodes, edges)
    return nodes, edges


def build_graph(current: Value):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges= traverse(current)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label=f'{n.name} | data: {n.data: .3f} | grad: {n.grad: .3f}', shape='record')

        if n._op:
            dot.node(name=uid+n._op, label=n._op)
            dot.edge(uid+n._op, uid)

    for node1, node2 in edges:
        dot.edge(str(id(node1)), str(id(node2))+node2._op)

    return dot