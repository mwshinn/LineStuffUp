from . import base as transform
import numpy as np
from . import ndarray_shifted as ndarray_shifted
from . import utils
import os
import tempfile
import sqlite3


class TransformGraph:
    def __init__(self, name):
        # NOTE: If you change the constructor or internal data structure, you also need to change the load and save methods.
        self.name = name
        self.nodes = [] # List of node names
        self.edges = {} # Dictionary of dictonaries, edges[node1][node2] = transform
        
        # node_images is a cache. It can contain:
        # - an ndarray if the image is loaded.
        # - None if the image exists in the DB but is not loaded.
        # - a 'ref:other_node' string if it's a reference to another node.
        # Keys are all nodes that have an associated image.
        self.node_images = {} 

        # compressed_node_images stores "dirty" images that need to be saved to the database.
        # Format: {node_name: (compressed_data, info)} for image data
        #      or {node_name: (ref_node_name, [])} for a reference
        self.compressed_node_images = {}
        
        self.node_notes = {}
        self.filename = None
        self.metadata = None

    def __eq__(self, other):
        # NOTE: This equality check does not compare image data for performance reasons.
        # It only checks if the same nodes have images.
        return (isinstance(other, TransformGraph) and
                self.name == other.name and
                self.nodes == other.nodes and
                self.edges == other.edges and
                set(self.node_images.keys()) == set(other.node_images.keys()))
    def __getitem__(self, item):
        if isinstance(item, str) and item in self.nodes:
            return self.get_image(item)
        if isinstance(item, slice) and isinstance(item.start, str) and isinstance(item.stop, str) and item.step is None and item.start in self.nodes and item.stop in self.nodes:
            return self.get_transform(item.start, item.stop)
        raise ValueError(f"A graph cannot have the item '{item}'")
    def __setitem__(self, name, value):
        if isinstance(name, str):
            return self.add_node(name, image=value)
        if isinstance(name, slice) and isinstance(name.start, str) and isinstance(name.stop, str) and name.step is None:
            return self.add_edge(name.start, name.stop, value)
        raise ValueError(f"A graph cannot assign the item '{item}'")
        
    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.nodes
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            return item[0] in self.edges.keys() and item[1] in self.edges[item[0]].keys()
        raise ValueError(f"A graph cannot contain the item '{item}'")

    def save(self, filename=None):
        if filename is None:
            filename = self.filename
        if filename is None:
            raise ValueError("Filename must be provided to save.")
        if filename.endswith(".npz"):
            raise ValueError("Saving in npz format is no longer supported")
        if "." not in filename:
            filename = filename+".db"

        self.filename = filename
        
        con = sqlite3.connect(filename)
        cur = con.cursor()

        cur.execute("PRAGMA foreign_keys = ON;")

        cur.execute('''
            CREATE TABLE IF NOT EXISTS graph_properties (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                name TEXT PRIMARY KEY
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS node_images (
                node_name TEXT PRIMARY KEY,
                data BLOB,
                info TEXT,
                ref_node TEXT,
                FOREIGN KEY(node_name) REFERENCES nodes(name) ON DELETE CASCADE
            )
        ''')

        cur.execute("BEGIN")
        try:
            properties = {
                'name': self.name,
                'edges': repr(self.edges),
                'notes': repr(self.node_notes),
                'metadata': repr(self.metadata)
            }
            cur.executemany("INSERT OR REPLACE INTO graph_properties VALUES (?, ?)", properties.items())

            cur.execute("SELECT name FROM nodes")
            db_nodes = {row[0] for row in cur.fetchall()}
            current_nodes = set(self.nodes)

            nodes_to_delete = db_nodes - current_nodes
            if nodes_to_delete:
                cur.executemany("DELETE FROM nodes WHERE name = ?", [(n,) for n in nodes_to_delete])

            nodes_to_add = current_nodes - db_nodes
            if nodes_to_add:
                cur.executemany("INSERT OR IGNORE INTO nodes (name) VALUES (?)", [(n,) for n in nodes_to_add])

            cur.execute("SELECT node_name FROM node_images")
            db_image_nodes = {row[0] for row in cur.fetchall()}
            current_image_nodes = set(self.node_images.keys())
            
            image_entries_to_delete = db_image_nodes - current_image_nodes
            if image_entries_to_delete:
                 cur.executemany("DELETE FROM node_images WHERE node_name = ?", [(n,) for n in image_entries_to_delete])

            for node_name, compressed_value in self.compressed_node_images.items():
                if isinstance(compressed_value[0], str) and compressed_value[1] == []: # Reference node
                    ref_node = compressed_value[0]
                    cur.execute(
                        "INSERT OR REPLACE INTO node_images (node_name, data, info, ref_node) VALUES (?, NULL, NULL, ?)",
                        (node_name, ref_node)
                    )
                else:  # Actual image data
                    data, info = compressed_value
                    cur.execute(
                        "INSERT OR REPLACE INTO node_images (node_name, data, info, ref_node) VALUES (?, ?, ?, NULL)",
                        (node_name, data, str(info))
                    )
            
            con.commit()
            self.compressed_node_images.clear()

        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    @classmethod
    def load(cls, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No such file or directory: '{filename}'")
        if filename.endswith(".npz"):
            return cls._load_npz_and_convert(filename)
        return cls._load_sqlite(filename)
    @classmethod
    def _load_sqlite(cls, filename):
        con = sqlite3.connect(f'file:{filename}?mode=ro', uri=True)
        cur = con.cursor()
        try:
            cur.execute("SELECT value FROM graph_properties WHERE key = 'name'")
            name = cur.fetchone()[0]
            g = cls(name)
            g.filename = filename

            cur.execute("SELECT key, value FROM graph_properties")
            props = dict(cur.fetchall())

            g.edges = eval(props['edges'], transform.__dict__, transform.__dict__)
            g.node_notes = eval(props.get('notes', '{}'))
            g.metadata = eval(props.get('metadata', 'None'))

            cur.execute("SELECT name FROM nodes")
            g.nodes = [row[0] for row in cur.fetchall()]

            cur.execute("SELECT node_name, ref_node FROM node_images")
            for node_name, ref_node in cur.fetchall():
                if ref_node is not None:
                    g.node_images[node_name] = f"ref:{ref_node}"
                else:
                    g.node_images[node_name] = None
        finally:
            con.close()
        return g

    @classmethod
    def _load_npz_and_convert(cls, filename):
        print(f"Loading legacy NPZ file: {filename}. It will be converted to the new SQLite format.")
        f = np.load(filename, allow_pickle=True)
        g = cls(str(f['name']))

        g.nodes = list(map(str, f['nodes']))
        g.edges = eval(str(f['edges']), transform.__dict__, transform.__dict__)
        if "notes" in f.keys():
            g.node_notes = eval(str(f['notes']))
        if "metadata" in f.keys():
            g.metadata = eval(str(f['metadata']))

        node_image_keys = f.get('nodeimage_keys', [])
        for i, n_bytes in enumerate(node_image_keys):
            n = str(n_bytes)
            compressed_value = (f[f'nodeimage_{i}'], f[f'nodeimageinfo_{i}'])

            info_obj = compressed_value[1]
            try:
                # Handle old format where string reference info was an empty ndarray
                is_ref = info_obj.size == 0
            except AttributeError:
                is_ref = False

            if is_ref:
                ref_node_name = str(compressed_value[0])
                g.compressed_node_images[n] = (ref_node_name, [])
                g.node_images[n] = f"ref:{ref_node_name}"
            else:
                data = compressed_value[0]
                info = list(compressed_value[1])
                g.compressed_node_images[n] = (data, [i.item() for i in info]) # Avoid numpy printing datatypes
                g.node_images[n] = None

        g.filename = os.path.splitext(filename)[0] + '.db'
        return g
    
    def add_node(self, name, image=None, compression="normal", notes=""):
        # Image can either be a 3-dimensional ndarray or a string of another node
        assert name not in self.nodes, f"Node '{name}' already exists"
        if image is not None:
            if isinstance(image, str):
                assert image in self.nodes, f"Referenced node '{image}' for new node '{name}' does not exist"
                self.compressed_node_images[name] = (image, [])
                self.node_images[name] = f"ref:{image}"
            else:
                if image.ndim == 2:
                    image = image[None]
                compressed = utils.compress_image(image, level=compression)
                self.compressed_node_images[name] = compressed
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
        del self.edges[name]
        for n in self.nodes:
            if n in self.edges and name in self.edges[n]:
                del self.edges[n][name]
        self.nodes.remove(name)

    def replace_node_image(self, name, image=None, compression="normal"):
        """Replace or remove a node's image without impacting its other connections"""
        assert name in self.nodes, f"Node '{name}' doesn't exist"

        if image is None:
            if name in self.node_images:
                del self.node_images[name]
            if name in self.compressed_node_images:
                del self.compressed_node_images[name]
            return
        
        if isinstance(image, str):
            assert image in self.nodes, f"Referenced node '{image}' for node '{name}' does not exist"
            self.compressed_node_images[name] = (image, [])
            self.node_images[name] = f"ref:{image}"
        else:
            if image.ndim == 2:
                image = image[None]
            compressed = utils.compress_image(image, level=compression)
            self.compressed_node_images[name] = compressed
            self.node_images[name] = image

    def add_edge(self, frm, to, transform, update=False):
        assert frm in self.nodes, f"Node '{frm}' doesn't exist"
        assert to in self.nodes, f"Node '{to}' doesn't exist"
        if update is False:
            assert to not in self.edges[frm].keys(), "Edge already exists"
        else:
            assert to in self.edges[frm].keys(), "Edge doesn't exist"
        self.edges[frm][to] = transform
        # At some point in the future we can support directed graphs, i.e.,
        # graphs with non-invertible transforms.
        inv = transform.invert()
        self.edges[to][frm] = inv

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
        # Before deleting nodes from node_images, make sure that this isn't
        # unsaved.
        nodes_to_unload = [
            node for node, image in self.node_images.items()
            if isinstance(image, np.ndarray) and node not in self.compressed_node_images
        ]
        for node_name in nodes_to_unload:
            self.node_images[node_name] = None
    
    def get_transform(self, frm, to):
        assert frm in self.nodes, f"Node {frm} not found"
        assert to in self.nodes, f"Node {to} not found"
        if frm == to:
            return transform.Identity()
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
        if node not in self.nodes:
            raise KeyError(f"Node '{node}' does not exist.")
        if node not in self.node_images:
            raise KeyError(f"Node '{node}' does not have an associated image.")

        # First try to load it from the cache
        cached_value = self.node_images[node]
        if isinstance(cached_value, np.ndarray):
            return cached_value
        if isinstance(cached_value, str) and cached_value.startswith('ref:'):
            imnode = cached_value.split(':', 1)[1]
            transformed_image = self.get_transform(imnode, node).transform_image(self.get_image(imnode), relative=True)
            self.node_images[node] = transformed_image
            return transformed_image
        # If it is not cached, we will need to decompress.
        if cached_value is None:
            # Look for the compressed image in the dirty images, and if you
            # can't find it there, then go to the db file on the disk.
            if node in self.compressed_node_images.keys():
                compressed_image = self.compressed_node_images[node]
            else:
                if not self.filename or not os.path.exists(self.filename):
                    raise RuntimeError("Graph has no associated database file to load image from.")

                con = sqlite3.connect(f'file:{self.filename}?mode=ro', uri=True)
                cur = con.cursor()
                try:
                    cur.execute("SELECT data, info FROM node_images WHERE node_name = ? AND ref_node IS NULL", (node,))
                    row = cur.fetchone()
                    if row is None:
                        raise RuntimeError(f"Image for node '{node}' not found in database '{self.filename}'.")
                    compressed_image = (row[0], eval(row[1]))
                finally:
                    con.close()
            data_bytes, info = compressed_image
            np_data = np.frombuffer(data_bytes, dtype=np.uint8)
            image = utils.decompress_image(np_data, info)
            self.node_images[node] = image
            return image
        raise RuntimeError(f"Internal error in get_image for node '{node}'. Invalid cache state: {cached_value}")

    def visualise(self, filename=None, nearby=None):
        fn = filename
        if fn is None:
            fn = tempfile.mkstemp()[1]
        try:
            import graphviz
        except ImportError:
            raise ImportError("Please install graphviz package to visualise")
        g = graphviz.Digraph(self.name, filename=fn)
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
        ur_nodes_used = set()
        for e1 in self.edges.keys():
            for e2 in self.edges[e1].keys():
                if nearby is not None and e1 != nearby and e2 != nearby:
                    continue
                if e1 in self.edges[e2].keys() and self.edges[e1][e2].__class__.__name__ == self.edges[e2][e1].__class__.__name__:
                    if e1 > e2 and self.edges[e1][e2].__class__.__name__ != "Identity":
                        g.edge(ur_node[e1], ur_node[e2], label=self.edges[e1][e2].__class__.__name__, dir="both")
                        ur_nodes_used.add(ur_node[e1])
                        ur_nodes_used.add(ur_node[e2])
                else:
                    g.edge(ur_node[e1], ur_node[e2], label=self.edges[e1][e2].__class__.__name__)
                    ur_nodes_used.add(ur_node[e1])
                    ur_nodes_used.add(ur_node[e2])
        for n in sorted(ur_nodes_used):
            g.node(n, label=ur_node_names[n])
        g.view()
        if filename is None: # Temporary file
            os.unlink(fn)
