from .base import Identity, Translate, Transform, PointTransform, AffineTransform
import numpy as np
import napari
import magicgui
import vispy
from . import utils
from .ndarray_shifted import ndarray_shifted

# Deprecated, functionality is in alignment_gui
def edit_transform(base_image, movable_image, transform):
    return alignment_gui(base_image, movable_image, transform_type=transform.__class__, initial_movable_points=transform.points_start, initial_base_points=transform.points_end)

def plot_from_graph(g, ims, output_space=None):
    """Given a graph g, plot each of the images in the list ims, using the first as a coordinate reference (base).

    Each element in ims can either be the name of a node on the graph (in which
    case the image from that node will be used, or a tuple of two elements where
    the first element is the image as an ndarray and the second element is the
    name of the node to start the transform (i.e., the coordinate space of the
    image).

    The list ims must be at least length 2.
    """
    if isinstance(ims[0], tuple):
        base_image = ims[0]
        base_space = ims[1]
    else:
        base_space = ims[0]
        base_image = g.get_image(base_space)
    if output_space is not None:
        base_image = g.get_transform(base_space, output_space).transform_image(base_image, labels=utils.image_is_label(base_image), relative=True)
        base_space = output_space
    els = []
    for im in ims[1:]:
        if isinstance(im, tuple):
            els.append((im[0], g.get_transform(im[1], base_space)))
        else:
            gi = g.get_image(im)
            if gi.shape[0] == 1:
                gi = gi * np.ones((2,1,1)) # TODO Hack for now when we can't see 1-plane images in napari
            els.append((gi, g.get_transform(im, base_space)))
    references = []
    if len(els) > 1:
        references = els[1:]
    alignment_gui(base_image, els[0][0], els[0][1], references=references)

def alignment_gui(base_image, movable_image, transform_type=Translate, initial_base_points=None, initial_movable_points=None, references=[]):
    """Align images

    If `base_image` and/or `movable_image` are tuples, they will be interpreted
    as multi-channel

    Reference should be a list of tuples, where each tuple is (image, transform)

    """
    if not isinstance(base_image, tuple):
        base_image = (base_image,)
    if not isinstance(movable_image, tuple):
        movable_image = (movable_image,)
    pretransform = transform_type.pretransform()
    tform = pretransform
    # Test if we are editing an existing transform
    if isinstance(transform_type, Transform) and initial_base_points is None and initial_movable_points is None:
        if isinstance(transform_type, PointTransform):
            initial_movable_points = transform_type.points_start
            initial_base_points = transform_type.points_end
        params = transform_type.params.copy()
        transform_type = transform_type.__class__
    else:
        params = transform_type.DEFAULT_PARAMETERS.copy()
    is_point_transform = issubclass(transform_type, PointTransform)
    if not is_point_transform:
        assert initial_base_points is None
        assert initial_movable_points is None
    _prev_matrix = None # A special case optimisation for linear transforms
    v = napari.Viewer()
    # v.window._qt_viewer._dockLayerList.setVisible(False)
    # v.window._qt_viewer._dockLayerControls.setVisible(False)
    base_points = [] if initial_base_points is None else list(initial_base_points)
    movable_points = [] if initial_movable_points is None else list(initial_movable_points)
    tform_type = transform_type
    layers_base = [v.add_image(bi, colormap="red", blending="additive", name="base", translate=(bi.origin if isinstance(bi, ndarray_shifted) else [0,0,0])) for bi in base_image]
    layers_movable = [v.add_image(tform.transform_image(mi, relative=True, labels=utils.image_is_label(mi)), colormap="blue", blending="additive", name="movable", translate=tform.origin_and_maxpos(mi)[0]) for mi in movable_image]
    layers_reference = [v.add_image(rt.transform_image(ri, relative=True, labels=utils.image_is_label(ri)), colormap="green", blending="additive", name=f"reference_{i}", translate=rt.origin_and_maxpos(ri)[0]) for i,(ri,rt) in enumerate(references)]
    if is_point_transform:
        layer_base_points = v.add_points(None, ndim=3, name="base points", edge_width=0, face_color=[1, .6, .6, 1])
        layer_movable_points = v.add_points(None, ndim=3, name="movable points", edge_width=0, face_color=[.6, .6, 1, 1])
        layer_base_points.data = base_points
        layer_movable_points.data = movable_points
        layer_base_points.editable = False
        layer_movable_points.editable = False
    def select_base_movable():
        # The logic to get this to work is out of order, so please read code in the
        # order specified in the comments.
        temp_points = []
        # Step 2: Processe base layer click
        def base_click_callback(viewer, e):
            if e.type != "mouse_press":
                return
            # Step 2a: Process base layer click
            temp_points.append(e.position)
            for layer_base in layers_base:
                layer_base.mouse_drag_callbacks.pop()
            for layer_movable in layers_movable:
                layer_movable.mouse_drag_callbacks.append(movable_click_callback)
            layer_base_points.data = np.vstack([layer_base_points.data, e.position])
            set_point_size()
            # Step 2b: Prepare for movable layer click
            v.layers.selection = set([layers_movable[0]])
            for layer_movable in layers_movable:
                layer_movable.opacity = 1
            for layer_base in layers_base:
                layer_base.opacity = .1
        # Step 3: Process movable layer click
        def movable_click_callback(viewer, e):
            nonlocal tform
            if e.type != "mouse_press":
                return
            # Step 3a: Process movable layer click
            base_points.append(temp_points[0])
            movable_points.append(pretransform.transform(tform.inverse_transform([e.position]))[0])
            for layer_movable in layers_movable:
                layer_movable.mouse_drag_callbacks.pop()
            for layer_base in layers_base:
                layer_base.opacity = 1
            # Step 3b: Clean up after clicks
            layer_base_points.data = base_points
            layer_movable_points.data = tform.transform(pretransform.inverse_transform(movable_points))
            set_point_size()
            v.layers.selection = prev_selection
            for b in buttons:
                b.enabled = True
        # Step 1: Wait for a click on the base layer
        v.layers.selection = set([layers_base[0]])
        for layer_movable in layers_movable:
            layer_movable.opacity = .1
        for layer_base in layers_base:
            layer_base.mouse_drag_callbacks.append(base_click_callback)
        prev_selection = v.layers.selection
        for b in buttons:
            b.enabled = False
    def remove_point():
        if len(base_points) == 0:
            return
        # The logic to get this to work is out of order, so please read code in the
        # order specified in the comments.
        temp_points = []
        # Step 2: Processe base layer click
        def remove_click_callback(viewer, e):
            if e.type != "mouse_press":
                return
            v.mouse_drag_callbacks.pop()
            # Step 2a: Find and remove the closest point (base or movable) to the click and its corresponding point (movable or base)
            search_point = e.position
            dists_base = np.sum(np.square(np.asarray(base_points) - [search_point]), axis=1)
            dists_movable = np.sum(np.square(np.asarray(tform.transform(pretransform.inverse_transform(movable_points))) - [search_point]), axis=1)
            ind = np.argmin(dists_base) if np.min(dists_base) < np.min(dists_movable) else np.argmin(dists_movable)
            base_points.pop(ind)
            movable_points.pop(ind)
            # Step 2b: Clean up
            for layer_base in layers_base:
                layer_base_points.data = base_points
            for layer_movable in layers_movable:
                layer_movable_points.data = tform.transform(pretransform.inverse_transform(movable_points))
            set_point_size()
            for b in buttons:
                b.enabled = True
            for layer_movable in layers_movable:
                layer_movable.opacity = 1
            for layer_base in layers_base:
                layer_base.opacity = 1
        # Step 1: Wait for a click on the base layer
        for layer_movable in layers_movable:
            layer_movable.opacity = .1
        for layer_base in layers_base:
            layer_base.opacity = .1
        for b in buttons:
            b.enabled = False
        v.mouse_drag_callbacks.append(remove_click_callback)
    def apply_transform(*args, transform=None, **kwargs):
        # kwargs here are extra parameters to pass to the transform.
        nonlocal tform, movable_points, params, _prev_matrix
        if transform is not None:
            tform = transform
            if is_point_transform:
                layer_movable_points.data = tform.transform(pretransform.inverse_transform(movable_points))
                layer_movable_points.refresh()
        elif is_point_transform:
            if movable_points is not None and len(movable_points) > 0:
                tform = tform_type(points_start=movable_points, points_end=base_points, **params)
            else:
                tform = pretransform
            layer_movable_points.data = tform.transform(pretransform.inverse_transform(movable_points))
            layer_movable_points.refresh()
        else:
            tform = tform_type(**params)
        for b in buttons: # Disable buttons while applying transform
            b.enabled = False
        for layer_movable,mi in zip(layers_movable,movable_image):
            # This if statement is a special case optimisation for
            # AffineTransforms only to avoid rerending the image if only the
            # origin/translation has changed.
            if _prev_matrix is None or (isinstance(tform, AffineTransform) and np.any(_prev_matrix != tform.matrix)):
                layer_movable.data = tform.transform_image(mi, relative=True, labels=utils.image_is_label(mi))
            layer_movable.translate = tform.origin_and_maxpos(mi)[0]
            layer_movable.refresh()
        if isinstance(tform, AffineTransform):
            _prev_matrix = tform.matrix
        for b in buttons: # Turn buttons back on when transform is done
            b.enabled = True
    def set_point_size(zoom=None):
        if zoom is None:
            zoom = v.camera.zoom
        if hasattr(zoom, "value"):
            zoom = zoom.value
        layer_base_points.size = 20/zoom
        layer_movable_points.size = 20/zoom
        layer_base_points.selected_data = []
        layer_movable_points.selected_data = []
        layer_base_points.refresh()
        layer_movable_points.refresh()
    v.layers.selection.clear()
    v.layers.selection.add(layers_base[0])
    button_add_point = magicgui.widgets.PushButton(value=True, text='Add new point')
    button_add_point.clicked.connect(select_base_movable)
    button_transform = magicgui.widgets.PushButton(value=True, text='Perform transform')
    button_transform.clicked.connect(apply_transform)
    button_reset = magicgui.widgets.PushButton(value=True, text='Reset transform')
    button_reset.clicked.connect(lambda : apply_transform(transform=pretransform))
    button_delete = magicgui.widgets.PushButton(value=True, text='Remove point')
    button_delete.clicked.connect(remove_point)
    if is_point_transform:
        buttons = [button_add_point, button_transform, button_reset, button_delete]
    else:
        buttons = [button_transform, button_reset]
    widgets = []
    widgets.extend(buttons)
    # For controlling parameters using the mouse
    _MOUSE_DRAG_WIDGETS = [None, None, None] # z, y, and x position widgets
    def mouse_drag_callback(viewer, event):
        if vispy.util.keys.CONTROL not in event.modifiers or vispy.util.keys.SHIFT not in event.modifiers:
            return
        if viewer.dims.ndisplay != 2:
            return
        initial_pos = [w.value if w is not None else 0 for w in _MOUSE_DRAG_WIDGETS]
        dd = event.dims_displayed
        base = event.position
        wh = event.source.size
        yield
        while event.type == "mouse_move":
            if _MOUSE_DRAG_WIDGETS[dd[0]] is not None:
                _MOUSE_DRAG_WIDGETS[dd[0]].value = event.position[dd[0]] - base[dd[0]] + initial_pos[dd[0]]
            if _MOUSE_DRAG_WIDGETS[dd[1]] is not None:
                _MOUSE_DRAG_WIDGETS[dd[1]].value = event.position[dd[1]] - base[dd[1]] + initial_pos[dd[1]]
            yield
    # Draw parameter spinboxes
    for p,pv in params.items():
        # This currently assumes all parameters are floats or bools
        if isinstance(pv, bool): # Bool
            w = magicgui.widgets.CheckBox(value=pv, label=p+":")
        else: # Float
            w = magicgui.widgets.FloatSpinBox(value=pv, label=p+":", min=-np.inf, max=np.inf)
        def widget_callback(*args,p=p,w=w):
            params[p] = w.value
            if dynamic_update.value:
                apply_transform()
        w.changed.connect(widget_callback)
        widgets.append(w)
        if p in transform_type.GUI_DRAG_PARAMETERS:
            _MOUSE_DRAG_WIDGETS[transform_type.GUI_DRAG_PARAMETERS.index(p)] = w
    dynamic_update = magicgui.widgets.CheckBox(value=False, label="Dynamic update")
    if len(params) > 0:
        widgets.append(dynamic_update)
    if not all(w is None for w in _MOUSE_DRAG_WIDGETS):
        v.mouse_drag_callbacks.append(mouse_drag_callback)
        dynamic_update.value = True
        widgets.insert(-1, magicgui.widgets.Label(value="Ctrl+Shift mouse drag to edit"))
    container_widget = magicgui.widgets.Container(widgets=widgets)
    v.window.add_dock_widget(container_widget, area="left", add_vertical_stretch=False)
    if is_point_transform:
        v.camera.events.zoom.connect(set_point_size)
        set_point_size()
    apply_transform()
    v.show(block=True)
    print(tform)
    return tform
