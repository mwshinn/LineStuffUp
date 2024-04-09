from . import base as transform
import numpy as np

class TransformGraph:
    def __init__(self, name):
        # NOTE: If you change the constructor or internal data structure, you also need to change the load and save methods.
        self.name = name
        self.nodes = [] # List of node names
        self.edges = {} # Dictionary of dictonaries, edges[node1][node2] = transform
        self.node_images = {} # If node has an associated image, node name is key and image is value
    def __eq__(self, other):
        return (self.name == other.name) and \
            self.nodes == other.nodes and \
            self.edges == other.edges and \
            len(self.node_images) == len(other.node_images) and \
            all(np.allclose(self.node_images[ni1],other.node_images[ni2]) for ni1,ni2 in zip(self.node_images.keys(), other.node_images.keys()))
    def save(self, filename):
        # Note to future self: If I ende up not using image arrays, I could rewrite this to save in text format.
        node_images_keys = list(self.node_images.keys())
        node_images_values = [self.node_images[k] for k in node_images_keys]
        node_images_arrays = {f"nodeimage_{i}": node_images_values[i] for i in range(0, len(node_images_values))}
        np.savez_compressed(filename, name=self.name, nodes=self.nodes, nodeimage_keys=node_images_keys, **node_images_arrays, edges=repr(self.edges))
    @classmethod
    def load(cls, filename):
        f = np.load(filename)
        g = cls(str(f['name']))
        g.nodes = list(map(str, f['nodes']))
        g.edges = eval(str(f['edges']), transform.__dict__, transform.__dict__)
        for i,n in enumerate(f['nodeimage_keys']):
            n = str(n)
            g.node_images[n] = f[f'nodeimage_{i}']
        return g
    def add_node(self, name, image=None):
        assert name not in self.nodes, f"Node '{name}' already exists"
        self.nodes.append(name)
        self.edges[name] = {}
        if image is not None:
            self.node_images[name] = image
    def add_edge(self, frm, to, transform):
        assert frm in self.nodes, "Node '{frm}' doesn't exist"
        assert to in self.nodes, "Node '{to}' doesn't exist"
        assert to not in self.edges[frm].keys(), "Edge already exists"
        self.edges[frm][to] = transform
        try:
            inv = transform.invert()
            assert frm not in self.edges[to].keys(), "Inverse edge already exists"
            self.edges[to][frm] = inv
        except NotImplementedError:
            pass
    def get_transform(self, frm, to):
        def _get_transform_from_chain(chain):
            cur = frm
            tform = None
            for c in chain:
                tform = self.edges[cur][c] if tform is None else tform + self.edges[cur][c]
                cur = c
            return tform
        candidates = list(map(tuple, self.edges[frm].keys()))
        while len(candidates) > 0:
            if to in [l[-1] for l in candidates]:
                chain = next(l for l in candidates if to == l[-1])
                return _get_transform_from_chain(chain)
            c0 = candidates.pop(0)
            to_append = [tuple(list(c0)+[n]) for n in self.edges[c0[-1]] if n not in c0]
            candidates.extend(to_append)
        raise RuntimeError(f"Path from '{frm}' to '{to}' not found")
    def get_image(self, node):
        return self.node_images[node]
