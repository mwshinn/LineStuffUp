# These tests are AI generated so I don't know if they are actually useful.
import unittest
import os
import shutil
import tempfile
import numpy as np

# Import the class to be tested
from graphsqlite import TransformGraph
from transform import Identity, TranslateFixed, PointTransformNoInverse, utils
import transform

class InvertibleError(PointTransformNoInverse):
    DEFAULT_PARAMETERS = {"extent": 1, "invert": False}
    def _transform(self, points, points_start, points_end):
        return points
    def invert(self, *args, **kwargs):
        raise NotImplementedError

class TestTransformGraph(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_graph.sqlite3")
        self.npz_path = os.path.join(self.test_dir, "old_graph.npz")

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def _create_sample_image(self, value=1, shape=(1, 10, 10)):
        """Helper to create a simple numpy array image."""
        return np.full(shape, value, dtype=np.uint8)

    def test_01_initialization(self):
        """Test basic graph initialization."""
        g = TransformGraph("MyTestGraph")
        self.assertEqual(g.name, "MyTestGraph")
        self.assertEqual(g.nodes, [])
        self.assertEqual(g.edges, {})
        self.assertEqual(g.node_images, {})

    def test_02_add_and_remove_node(self):
        """Test adding and removing nodes."""
        g = TransformGraph("NodeTest")
        img1 = self._create_sample_image(1)
        
        # Add a node with an image
        g.add_node("node1", image=img1, notes="This is node 1.")
        self.assertIn("node1", g.nodes)
        self.assertIn("node1", g.edges)
        self.assertIn("node1", g.node_images)
        self.assertIn("node1", g.compressed_node_images) # Should be dirty
        self.assertEqual(g.node_notes["node1"], "This is node 1.")
        np.testing.assert_array_equal(g.get_image("node1"), img1)

        # Add a node without an image
        g.add_node("node2")
        self.assertIn("node2", g.nodes)
        self.assertNotIn("node2", g.node_images)
        with self.assertRaises(KeyError):
            g.get_image("node2")
            
        # Add a node that references another
        g.add_node("node1_ref", image="node1")
        self.assertIn("node1_ref", g.node_images)
        self.assertEqual(g.node_images['node1_ref'], 'ref:node1')

        # Remove the node
        g.remove_node("node1")
        self.assertNotIn("node1", g.nodes)
        self.assertNotIn("node1", g.edges)
        self.assertNotIn("node1", g.node_images)
        self.assertNotIn("node1", g.compressed_node_images)
        self.assertNotIn("node1", g.node_notes)

    def test_03_add_and_remove_edge(self):
        """Test adding and removing edges."""
        g = TransformGraph("EdgeTest")
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")

        # Add an invertible edge
        t_ab = TranslateFixed(x=10)
        g.add_edge("A", "B", t_ab)
        self.assertIn("B", g.edges["A"])
        self.assertIs(g.edges["A"]["B"], t_ab)
        # Check for automatic inverse
        self.assertIn("A", g.edges["B"])
        self.assertIsInstance(g.edges["B"]["A"], TranslateFixed)

        # Add a non-invertible edge
        t_ac = InvertibleError()
        g.add_edge("A", "C", t_ac)
        self.assertIn("C", g.edges["A"])
        self.assertNotIn("A", g.edges["C"]) # No inverse

        # Remove edge
        g.remove_edge("A", "B")
        self.assertNotIn("B", g.edges["A"])
        self.assertNotIn("A", g.edges["B"])

    def test_04_save_and_load_sqlite(self):
        """Test saving to and loading from a SQLite database."""
        g_orig = TransformGraph("DBSaveLoadTest")
        img1 = self._create_sample_image(10)
        img2 = self._create_sample_image(20)

        g_orig.add_node("n1", image=img1, notes="Note for n1")
        g_orig.add_node("n2", image=img2)
        g_orig.add_node("n3", image="n1") # Reference node
        g_orig.add_node("n4")
        g_orig.add_edge("n1", "n2", TranslateFixed(x=5))
        g_orig.metadata = {"author": "tester"}

        g_orig.save(self.db_path)
        self.assertTrue(os.path.exists(self.db_path))

        # Load into a new graph object
        g_loaded = TransformGraph.load(self.db_path)

        self.assertEqual(g_orig.name, g_loaded.name)
        self.assertEqual(sorted(g_orig.nodes), sorted(g_loaded.nodes))
        self.assertEqual(g_orig.node_notes, g_loaded.node_notes)
        self.assertEqual(g_orig.metadata, g_loaded.metadata)
        self.assertEqual(repr(g_orig.edges), repr(g_loaded.edges))
        
        # Test dynamic image loading
        self.assertIsNone(g_loaded.node_images["n1"]) # Should not be loaded yet
        np.testing.assert_array_equal(g_loaded.get_image("n1"), img1)
        self.assertIsNotNone(g_loaded.node_images["n1"]) # Should be cached now
        
        np.testing.assert_array_equal(g_loaded.get_image("n2"), img2)
        
        # Test loading of a referenced image
        self.assertEqual(g_loaded.node_images["n3"], "ref:n1")
        # get_image should calculate the transformed image
        transformed_img = TranslateFixed(x=0).transform_image(img1, relative=True) # Bogus transform, just for check
        # With our mocks, the path n1->n3 is empty, so there will be an error
        with self.assertRaises(RuntimeError):
            g_loaded.get_image("n3")

    def test_05_selective_image_saving(self):
        """Test that only modified images are saved."""
        g = TransformGraph("SelectiveSave")
        img1 = self._create_sample_image(1)
        img2 = self._create_sample_image(2)

        g.add_node("node1", image=img1)
        g.add_node("node2", image=img2)
        
        # Initial save, all images are dirty
        self.assertIn("node1", g.compressed_node_images)
        self.assertIn("node2", g.compressed_node_images)
        g.save(self.db_path)
        self.assertEqual(g.compressed_node_images, {}) # Dirty dict should be cleared

        # Load, modify one image, and resave
        g_loaded = TransformGraph.load(self.db_path)
        self.assertEqual(g_loaded.compressed_node_images, {})

        new_img2 = self._create_sample_image(99)
        g_loaded.replace_node_image("node2", new_img2)
        
        # Now only node2 should be dirty
        self.assertNotIn("node1", g_loaded.compressed_node_images)
        self.assertIn("node2", g_loaded.compressed_node_images)

        g_loaded.save() # Resave to the same path
        self.assertEqual(g_loaded.compressed_node_images, {})

        # Load again and verify changes
        g_final = TransformGraph.load(self.db_path)
        np.testing.assert_array_equal(g_final.get_image("node1"), img1)
        np.testing.assert_array_equal(g_final.get_image("node2"), new_img2)

    def test_06_npz_backward_compatibility(self):
        """Test loading an old .npz file and converting it."""
        # Create a fake old .npz file
        name = "OldNPZ"
        nodes = ["A", "B"]
        edges = repr({'A': {'B': TranslateFixed(x=1)}, 'B': {'A': TranslateFixed(x=1)}})
        node_images_keys = ["A"]
        # Use our mock compression to get the right format
        img_a_data, img_a_info = utils.compress_image(self._create_sample_image(50))

        np.savez_compressed(
            self.npz_path, 
            name=name, 
            nodes=nodes, 
            edges=edges, 
            nodeimage_keys=node_images_keys, 
            nodeimage_0=img_a_data, 
            nodeimageinfo_0=img_a_info
        )

        # Load the .npz file
        g = TransformGraph.load(self.npz_path)

        # Check for conversion
        new_db_path = os.path.join(self.test_dir, "old_graph.sqlite3")
        self.assertTrue(os.path.exists(new_db_path))
        self.assertEqual(g.filename, new_db_path)
        
        # Check data integrity
        self.assertEqual(g.name, name)
        self.assertEqual(sorted(g.nodes), sorted(nodes))
        np.testing.assert_array_equal(g.get_image("A"), self._create_sample_image(50))

    def test_07_get_transform(self):
        """Test pathfinding for transforms."""
        g = TransformGraph("PathTest")
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_node("D") # Disconnected
        
        g.add_edge("A", "B", TranslateFixed(x=10))
        g.add_edge("B", "C", TranslateFixed(x=5))

        # Direct transform
        t_ab = g.get_transform("A", "B")
        self.assertIsInstance(t_ab, TranslateFixed)
        self.assertEqual(t_ab, TranslateFixed(x=10))

        # Chained transform
        t_ac = g.get_transform("A", "C")
        self.assertTrue(np.all(t_ac.transform([1, 2, 3]) == transform.TranslateFixed(x=15).transform([1, 2, 3])))
        
        # Identity transform
        t_aa = g.get_transform("A", "A")
        self.assertIsInstance(t_aa, Identity)

        # Path not found
        with self.assertRaises(RuntimeError):
            g.get_transform("A", "D")

    def test_08_unload(self):
        """Test unloading images from the cache."""
        g = TransformGraph("UnloadTest")
        img1 = self._create_sample_image(1)
        g.add_node("node1", image=img1)
        g.save(self.db_path)

        g_loaded = TransformGraph.load(self.db_path)
        # Load image into cache
        g_loaded.get_image("node1")
        self.assertIsInstance(g_loaded.node_images["node1"], np.ndarray)

        # Unload
        g_loaded.unload()
        self.assertIsNone(g_loaded.node_images["node1"])

        # Getting it again should reload from DB
        np.testing.assert_array_equal(g_loaded.get_image("node1"), img1)
    
    def test_09_remove_node_with_image_from_db(self):
        """Test that removing a node also removes its image on save."""
        g = TransformGraph("RemoveFromDB")
        g.add_node("n1", image=self._create_sample_image(1))
        g.add_node("n2", image=self._create_sample_image(2))
        g.save(self.db_path)
        
        # Load, remove a node, and save again
        g_loaded = TransformGraph.load(self.db_path)
        g_loaded.remove_node("n1")
        g_loaded.save()
        
        # Load final version and check
        g_final = TransformGraph.load(self.db_path)
        self.assertIn("n2", g_final.nodes)
        self.assertNotIn("n1", g_final.nodes)
        self.assertNotIn("n1", g_final.node_images)
        with self.assertRaises(KeyError):
            g_final.get_image("n1")
        
        # Verify the image for n2 is still there
        np.testing.assert_array_equal(g_final.get_image("n2"), self._create_sample_image(2))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
