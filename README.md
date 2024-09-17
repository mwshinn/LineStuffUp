To install, you need the following packages:

> pip install numpy scipy napari magicgui vispy scikit-image imageio imageio-ffmpeg

# Conceptual summary

## Transforms

A *Transform* takes you from one coordinate space to another coordinate space.
For instance, suppose you have a volumetric image , and a second volumetric
image rescaled to have uniform voxel size of 1um.  A Transform could map points
or images between the raw and rescaled coordinate spaces.

There are many types of Transforms included by default in this library.  These
fall into two main categories:

- *Parameter-based Transforms* use parametric values to define the Transform.
  For instance, TranslationFixed is a parameter-based Transform that receives an
  explicit z, y, and x shift.
- *Point-based Transforms* use a point cloud to define the Transform.  For
  point-based Transforms, you must define the starting and ending positions of
  several keypoints.  For instance, a Translation will find the z, y, and x
  shifts that best fit the keypoints.  You can choose these keypoints from a
  gui.  Some point-based Transforms may also include parameters, such as a
  smoothness hyperparameter or a normal vector along which the Transform should
  occur.

Transforms are invertible.  You can use the ``.invert()`` function to perform
the inversion.  This occurs analytically for most Transforms, but for some
non-rigid Transforms (e.g. DistanceWeightedAverage), the inverse Transform will
be much (1000x or more) slower than the forward Transform.

Transforms may be specified or unspecified.  A specified Transform includes
values for each of its parameters, and matching point clouds if it is a
Point-based Transform.  This is represented by an instance of the class.  An
unspecified Transform does not yet have chosen parameters or points, and is
represented by the uninsantiated class.  For instance,
``TranslationFixed(x=3,y=0,z=1)`` is specified, but ``TranslationFixed`` is
unspecified.

Transforms are composable.  If you have two Transforms, you can add them
together to get their composition.  Two specified Transforms may be composed,
and their composition gives another specified Transform.

A specified and unspecified Transform may be composed.  Their composition gives
an unspecified transform.  Currently, the unspecified Transform must be the
final term in the sum (but this limitation may be lifted in future versions).
Two unspecified Transforms cannot be composed.

Transforms may be fit using ``transform.gui.alignment_gui``.  This takes a tuple
of images in the input space, a tuple of images in the output space, and an
unspecified Transform that you would like to fit.  If the unspecified Transform
is a composition between an unspecified and specified Transform, the input image
will first be first transformed according to the specified Transform, and then
you will be able to fit the unspecified part of the transform.  If a specified
Transform is passed instead of an unspecified Transform, you will be able to
edit the Transform, or, if it is a composition, the final step of the composed
Transform.

All information needed to save a Transform comes from its text representation.
So, you can simply call "print" and then copy and paste it somewhere, or save
the text of the Transform to a text file.  The string representation is
executable Python code that you can run to recreate your Transform.

## Graphs

When transforming across multiple different spaces and using several different
images, many Transforms will be needed.  It can quickly become difficult to
manage which Transform takes you from which space to which other space.  We can
organise all of these Transforms into a TransformGraph.

A TransformGraph is an undirected graph of Transforms from each space to each
other space.  Each space (e.g., image) is identified by a unique name, and is
represented by a node in the graph.  Each edge connecting the nodes in the graph
is a Transform.  To create a new node in a TransformGraph ``g``, run
``g.add_node(node_name)``.  To specify a Transform between two nodes, i.e., an
edge, run ``g.add_edge(node1, node2, tform)``.

This library always uses the "from -> to" convention in the order of arguments.
So in the previous example, the Transform ``tform`` converts points in space
``node1`` to the space ``node2``.  Or, equivalently, "movable image -> base
image", where ``node1`` is the movable image and ``node2`` is the base image.

To obtain the transform between two nodes, use the function
``g.get_transform(node1, node2)``.  Even if ``node1`` and ``node2`` are not
directly connected, the shortest path of Transform compositions will be computed
and returned.  If two nodes have no connection, this will raise an error.

To visualise the structure of the graph, run ``g.visualise()``.  For extremely
large graphs, you can use the "nearby" argument to specify a node, and the
visualisation will only include nodes directly connected to the given node.

Optionally, a TransformGraph may also contain the raw images themselves.  This
is accomplished by passing the "image" argument to ``g.add_node``.  The images
will be aggressively compressed with minimal loss in quality through the use of
video codecs, with compression rates on high-resolution microscopy images often
approaching 100:1.

When images are included directly, several convenience methods can be used.
Most notably, the ``GraphViewer`` is a napari viewer that accepts node names as
image or label layers.  The base coordinate system is the first added image, and
all subsequent added images will be transformed into the space of first image.
If there is no path of Transforms in the graph, adding the other images will
return an error.  Additionally, it allows using ``graph_alignment_gui``, a
shortcut version of ``alignment_gui`` that accepts node names instead of images.

## Shifted NDArrays

Normally you should not encounter Shifted NDArrays.  This is an internal data
storage which adds a origin offset to an NDArray.  This allows efficient
representation and modification of images which undergoes translation relative
to another image.
