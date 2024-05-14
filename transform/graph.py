from . import base as transform
import numpy as np
from . import ndarray_shifted as ndarray_shifted
from . import utils

class TransformGraph:
    def __init__(self, name):
        # NOTE: If you change the constructor or internal data structure, you also need to change the load and save methods.
        self.name = name
        self.nodes = [] # List of node names
        self.edges = {} # Dictionary of dictonaries, edges[node1][node2] = transform
        self.node_images = {} # If node has an associated image, node name is key and image is value
        self.compressed_node_images = {} # If a node has an associated image, the compressed version is stored here and loaded dynamically into node_images
        self.node_notes = {}
    def __eq__(self, other):
        return (self.name == other.name) and \
            self.nodes == other.nodes and \
            self.edges == other.edges and \
            len(self.compressed_node_images) == len(other.compressed_node_images) and \
            all(np.allclose(self.compressed_node_images[ni1][0],other.node_images[ni2][0]) for ni1,ni2 in zip(self.node_images.keys(), other.node_images.keys())) and \
            all(np.allclose(self.node_images[ni1][1],other.node_images[ni2][1]) for ni1,ni2 in zip(self.node_images.keys(), other.node_images.keys()))
    def save(self, filename):
        # Note to future self: If I ende up not using image arrays, I could rewrite this to save in text format.
        node_images_keys = list(sorted(self.compressed_node_images.keys()))
        node_images_values = [self.compressed_node_images[k] for k in node_images_keys]
        node_image_arrays_compressed = {f"nodeimage_{i}": node_images_values[i][0] for i in range(0, len(node_images_values))}
        node_image_arrays_info = {f"nodeimageinfo_{i}": node_images_values[i][1] for i in range(0, len(node_images_values))}
        np.savez_compressed(filename, name=self.name, nodes=self.nodes, nodeimage_keys=node_images_keys, **node_image_arrays_compressed, **node_image_arrays_info, edges=repr(self.edges), notes=repr(self.node_notes))
    @classmethod
    def load(cls, filename):
        f = np.load(filename)
        g = cls(str(f['name']))
        g.nodes = list(map(str, f['nodes']))
        g.edges = eval(str(f['edges']), transform.__dict__, transform.__dict__)
        for i,n in enumerate(f['nodeimage_keys']):
            n = str(n)
            g.compressed_node_images[n] = (f[f'nodeimage_{i}'], f[f'nodeimageinfo_{i}'])
        if "notes" in f.keys():
            g.node_notes = eval(str(f['notes']))
        return g
    @classmethod
    def load_old(cls, filename):
        f = np.load(filename)
        g = cls(str(f['name']))
        g.nodes = list(map(str, f['nodes']))
        g.edges = eval(str(f['edges']), transform.__dict__, transform.__dict__)
        for i,n in enumerate(f['nodeimage_keys']):
            n = str(n)
            g.node_images[n] = f[f'nodeimage_{i}']
        return g
    def add_node(self, name, image=None, compression="normal", notes=""):
        # Image can either be a 3-dimensional ndarray or a string of another node
        assert name not in self.nodes, f"Node '{name}' already exists"
        if image is not None: # Do this first because it may fail due to a memory error, and we don't want the node half-added
            if isinstance(image, str):
                self.compressed_node_images[name] = (image, [])
            else:
                if image.ndim == 2:
                    image = image[None]
                self.compressed_node_images[name] = utils.compress_image(image, level=compression)
                self.node_images[name] = image
        self.node_notes[name] = notes
        self.nodes.append(name)
        self.edges[name] = {}
    def add_edge(self, frm, to, transform):
        assert frm in self.nodes, f"Node '{frm}' doesn't exist"
        assert to in self.nodes, f"Node '{to}' doesn't exist"
        assert to not in self.edges[frm].keys(), "Edge already exists"
        self.edges[frm][to] = transform
        try:
            inv = transform.invert()
            assert frm not in self.edges[to].keys(), "Inverse edge already exists"
            self.edges[to][frm] = inv
        except NotImplementedError:
            pass
    def remove_edge(self, frm, to):
        assert frm in self.nodes, f"Node '{frm}' doesn't exist"
        assert to in self.nodes, f"Node '{to}' doesn't exist"
        assert to in self.edges[frm].keys(), "Edge doesn't exist"
        del self.edges[frm][to]
        if frm in self.edges[to].keys():
            del self.edges[to][frm]
    def unload(self):
        """Clear memory by unloading the node images, keeping only the compressed forms"""
        for k in self.node_images.keys():
            del self.node_images[k]
    def get_transform(self, frm, to):
        def _get_transform_from_chain(chain):
            cur = frm
            tform = None
            for c in chain:
                tform = self.edges[cur][c] if tform is None else tform + self.edges[cur][c]
                cur = c
            return tform
        candidates = list(map(lambda x : (x,) if isinstance(x, str) else tuple(x), self.edges[frm].keys()))
        while len(candidates) > 0:
            if to in [l[-1] for l in candidates]:
                chain = next(l for l in candidates if to == l[-1])
                return _get_transform_from_chain(chain)
            c0 = candidates.pop(0)
            to_append = [tuple(list(c0)+[n]) for n in self.edges[c0[-1]] if n not in c0]
            candidates.extend(to_append)
        raise RuntimeError(f"Path from '{frm}' to '{to}' not found")
    def get_image(self, node):
        if node not in self.node_images.keys():
            if len(self.compressed_node_images[node][1]) == 0: # First element is a string of a node
                imnode = str(self.compressed_node_images[node][0])
                self.node_images[node] = self.get_transform(imnode, node).transform_image(self.get_image(imnode), relative=True)
            else:
                self.node_images[node] = utils.decompress_image(*self.compressed_node_images[node])
        return self.node_images[node]
    def visualise(self, filename):
        try:
            import graphviz
        except ImportError:
            raise ImportError("Please install graphviz package to visualise")
        g = graphviz.Digraph(self.name, filename=filename)
        for e1 in self.edges.keys():
            for e2 in self.edges[e1].keys():
                if e1 in self.edges[e2].keys() and self.edges[e1][e2].__class__.__name__ == self.edges[e2][e1].__class__.__name__:
                    if e1 > e2:
                        g.edge(e1, e2, label=self.edges[e1][e2].__class__.__name__, dir="both")
                else:
                    g.edge(e1, e2, label=self.edges[e1][e2].__class__.__name__)
        g.view()
