from . import base as transform
import numpy as np
from . import ndarray_shifted as ndarray_shifted
from . import utils
import os
import tempfile

class TransformGraph:
    def __init__(self, name):
        # NOTE: If you change the constructor or internal data structure, you also need to change the load and save methods.
        self.name = name
        self.nodes = [] # List of node names
        self.edges = {} # Dictionary of dictonaries, edges[node1][node2] = transform
        self.node_images = {} # If node has an associated image, node name is key and image is value
        self.compressed_node_images = {} # If a node has an associated image, the compressed version is stored here and loaded dynamically into node_images
        self.node_notes = {}
        self.filename = None
    def __eq__(self, other):
        return (self.name == other.name) and \
            self.nodes == other.nodes and \
            self.edges == other.edges and \
            len(self.compressed_node_images) == len(other.compressed_node_images) and \
            all(np.allclose(self.compressed_node_images[ni1][0],other.compressed_node_images[ni2][0]) for ni1,ni2 in zip(self.compressed_node_images.keys(), other.compressed_node_images.keys()))
    def save(self, filename=None):
        if filename is None:
            filename = self.filename
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
        g.filename = filename
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
        # TODO this doesn't handle the case where other node images refer to the given node
    def remove_node(self, name):
        if name in self.compressed_node_images:
            del self.compressed_node_images[name]
        if name in self.node_images:
            del self.node_images[name]
        if name in self.node_notes:
            del self.node_notes[name]
        for n in list(self.edges[name]):
            del self.edges[name][n]
            if name in self.edges[n]:
                del self.edges[n][name]
        self.nodes.remove(name)
    def replace_node_image(self, name, image=None, compression="normal"):
        """Replace or remove a node's image without impacting its other connections"""
        # Mostly copied from add_node
        assert name in self.nodes, f"Node '{name}' doesn't exist"
        if name in self.node_images:
            del self.node_images[name]
        if image is not None: # Do this first because it may fail due to a memory error, and we don't want the node half-added
            if isinstance(image, str):
                self.compressed_node_images[name] = (image, [])
            else:
                if image.ndim == 2:
                    image = image[None]
                self.compressed_node_images[name] = utils.compress_image(image, level=compression)
                self.node_images[name] = image
        else:
            if name in self.compressed_node_images.keys():
                del self.compressed_node_images[name]
    def add_edge(self, frm, to, transform, update=False):
        assert frm in self.nodes, f"Node '{frm}' doesn't exist"
        assert to in self.nodes, f"Node '{to}' doesn't exist"
        if update is False:
            assert to not in self.edges[frm].keys(), "Edge already exists"
        else:
            assert to in self.edges[frm].keys(), "Edge doesn't exist"
        self.edges[frm][to] = transform
        try:
            inv = transform.invert()
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
    def connected_components(self):
        """Find connected components in the graph.

        This does not yet support directed graphs, i.e., graphs which contain
        non-invertable transforms.

        """
        components = []
        for n in self.nodes:
            # Make sure n isn't accounted for already
            if any([n in c for c in components]):
                continue
            # Find all nodes reachable from n and add to current_component.
            # Only search through those that haven't been searched through yet.
            current_component = set([n])
            to_search = [n]
            while len(to_search) > 0:
                node = to_search.pop()
                connected = list(self.edges[node].keys())
                to_search.extend([c for c in connected if c not in current_component])
                current_component = current_component.union(set(connected))
            components.append(current_component)
        return components
    def unload(self):
        """Clear memory by unloading the node images, keeping only the compressed forms"""
        keys = list(self.node_images.keys())
        for k in keys:
            del self.node_images[k]
    def get_transform(self, frm, to):
        assert frm in self.nodes, f"Node {frm} not found"
        assert to in self.nodes, f"Node {to} not found"
        def _get_transform_from_chain(chain):
            cur = frm
            tform = None
            for c in chain:
                tform = self.edges[cur][c] if tform is None else tform + self.edges[cur][c]
                cur = c
            return tform
        candidates = list(map(lambda x : (x,) if isinstance(x, str) else tuple(x), self.edges[frm].keys()))
        seen = [frm]
        while len(candidates) > 0:
            if to in [l[-1] for l in candidates]:
                chain = next(l for l in candidates if to == l[-1])
                return _get_transform_from_chain(chain)
            c0 = candidates.pop(0)
            seen.append(c0[-1])
            to_append = [tuple(list(c0)+[n]) for n in self.edges[c0[-1]] if n not in seen]
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
    def visualise(self, filename=None, nearby=None):
        fn = filename
        if fn is None:
            fn = tempfile.mkstemp()[1]
        try:
            import graphviz
        except ImportError:
            raise ImportError("Please install graphviz package to visualise")
        g = graphviz.Digraph(self.name, filename=filename)
        # Find all nodes that have an Identity edge and choose one as the 'base" node
        ur_node = {}
        ur_node_names = {}
        for e1 in self.edges.keys():
            found = False
            ident_edges = [e2 for e2 in self.edges[e1] if self.edges[e1][e2].__class__.__name__ == "Identity"]
            for e2 in ident_edges:
                if e2 in ur_node.keys() and ur_node[e2] == e2:
                    ur_node[e1] = e2
                    ur_node_names[e2] += "\n"+e1
                    found = True
                    break
            if not found:
                ur_node[e1] = e1
                ur_node_names[e1] = e1
        for e1 in self.edges.keys():
            for e2 in self.edges[e1].keys():
                if nearby is not None and e1 != nearby and e2 != nearby:
                    continue
                if e1 in self.edges[e2].keys() and self.edges[e1][e2].__class__.__name__ == self.edges[e2][e1].__class__.__name__:
                    if e1 > e2 and self.edges[e1][e2].__class__.__name__ != "Identity":
                        g.edge(ur_node[e1], ur_node[e2], label=self.edges[e1][e2].__class__.__name__, dir="both")
                else:
                    g.edge(ur_node[e1], ur_node[e2], label=self.edges[e1][e2].__class__.__name__)
        for n in ur_node_names.keys():
            g.node(n, label=ur_node_names[n])
        g.view()
        if filename is None: # Temporary file
            os.unlink(fn)
