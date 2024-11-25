from .base import Identity, Translate, Transform, PointTransform, AffineTransform
import numpy as np
import napari
import magicgui
import vispy
from . import utils
from .ndarray_shifted import ndarray_shifted

class GraphViewer(napari.Viewer):
    def __init__(self, graph, space=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "space", space)
        if isinstance(space, str):
            self.title = f"Alignment in {space} space"
    def _get_data_origin_name(self, data, space, labels=False):
        name = "data"
        if isinstance(data, str):
            name = data
            space = data if space is None else space
            data = self.graph.get_image(data)
        if self.space is None and space is not None: # First image sets the space if unset
            object.__setattr__(self, "space", space)
            self.title = f"Alignment in {space} space"
        if data.shape[0] == 1:
            data = data * np.ones((2,1,1), dtype="int") # TODO Hack for now when we can't see 1-plane images in napari
        if self.space is not None and space is not None and self.space != space:
            data = self.graph.get_transform(space, self.space).transform_image(data, labels=labels)
        origin = data.origin if isinstance(data, ndarray_shifted) else np.zeros_like(data.shape)
        scale = data.scale if isinstance(data, ndarray_shifted) else np.ones_like(data.shape)
        return data, origin, scale, name
    def add_image(self, data, space=None, **kwargs):
        data, origin, scale, name = self._get_data_origin_name(data, space)
        return super().add_image(data, translate=origin, name=name, scale=scale, **kwargs)
    def add_labels(self, data, space=None, **kwargs):
        data, origin, scale, name = self._get_data_origin_name(data, space, labels=True)
        return super().add_labels(data, translate=origin, name=name, scale=scale, **kwargs)
    def add_points(self, data, space=None, **kwargs):
        if space is not None and self.space is not None:
            data = self.graph.get_transform(space, self.space).transform(data)
        return super().add_points(data, **kwargs)

# Deprecated, functionality is in alignment_gui
def edit_transform(movable_image, base_image, transform):
    return alignment_gui(movable_image, base_image, transform_type=transform.__class__, initial_movable_points=transform.points_start, initial_base_points=transform.points_end)

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
    alignment_gui(els[0][0], base_image, els[0][1], references=references)

def graph_alignment_gui(g, movable, base, transform_type=None, add_transform=False, references=[]):
    # TODO Currently assumes all base images are in the same space
    if not isinstance(base, (list, tuple)):
        base = [base]
    if not isinstance(movable, (list, tuple)):
        movable = [movable]
    base_images = [g.get_image(bi) for bi in base]
    movable_images = [g.get_image(mi) for mi in movable]
    if transform_type is None:
        transform_type = g.get_transform(movable[0], base[0])
    elif add_transform:
        transform_type = g.get_transform(movable[0], base[0]) + transform_type
    if references:
        references = [(g.get_image(r), g.get_transform(r, base[0])) for r in references] 
    return alignment_gui(tuple(movable_images), tuple(base_images), transform_type=transform_type, references=references)

def alignment_gui(movable_image, base_image, transform_type=Translate, initial_base_points=None, initial_movable_points=None, downsample=None, references=[], crop=False, auto_find_peak_radius=2):
    """Align images

    If `base_image` and/or `movable_image` are tuples, they will be interpreted
    as multi-channel

    Reference should be a list of tuples, where each tuple is (image, transform)

    "crop" allows you to reduce the drawn area of the transformed image, making transforms faster and use less memory.

    """
    dsscale = np.ones(3) if downsample is None else np.asarray(downsample)
    if not isinstance(base_image, tuple):
        base_image = (base_image,)
    if not isinstance(movable_image, tuple):
        movable_image = (movable_image,)
    bi0 = ndarray_shifted(base_image[0])
    rel = True if crop is False else tuple(zip(bi0.origin, bi0.origin+bi0.shape)) if crop is True else crop
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
        print("Setting default params")
        params = transform_type.DEFAULT_PARAMETERS.copy()
    is_point_transform = issubclass(transform_type, PointTransform)
    if not is_point_transform:
        assert initial_base_points is None
        assert initial_movable_points is None
    _prev_matrix = None # A special case optimisation for linear transforms
    _prev_translate = None # A special case optimisation for linear transforms
    v = napari.Viewer()
    # v.window._qt_viewer._dockLayerList.setVisible(False)
    # v.window._qt_viewer._dockLayerControls.setVisible(False)
    base_points = [] if initial_base_points is None else list(initial_base_points)
    movable_points = [] if initial_movable_points is None else list(initial_movable_points)
    tform_type = transform_type
    layers_base = [v.add_image(bi, colormap="red", blending="additive", name="base", translate=(bi.origin if isinstance(bi, ndarray_shifted) else [0,0,0]), scale=(bi.scale if isinstance(bi, ndarray_shifted) else [1, 1, 1])) for bi in base_image]
    layers_movable = [v.add_image(tform.transform_image(mi, relative=rel, labels=utils.image_is_label(mi), downsample=downsample), colormap="green", blending="additive", name="movable", translate=tform.origin_and_maxpos(mi, relative=rel)[0], scale=downsample) for mi in movable_image]
    layers_reference = [v.add_image(rt.transform_image(ri, relative=rel, labels=utils.image_is_label(ri), downsample=downsample), colormap="blue", blending="additive", name=f"reference_{i}", translate=rt.origin_and_maxpos(ri, relative=rel)[0], scale=downsample) for i,(ri,rt) in enumerate(references)]
    if is_point_transform:
        layer_base_points = v.add_points(None, ndim=3, name="base points", edge_width=0, face_color=[1, .6, .6, 1])
        layer_movable_points = v.add_points(None, ndim=3, name="movable points", edge_width=0, face_color=[.6, 1, .6, 1])
        layer_base_points.data = base_points
        layer_movable_points.data = movable_points
        layer_base_points.editable = False
        layer_movable_points.editable = False
    def select_base_movable():
        # The logic to get this to work is out of order, so please read code in the
        # order specified in the comments.
        temp_points = []
        # Utility function: local ascent
        def find_local_maximum(image, starting_point, w=3):
            point = np.round(starting_point).astype(int)
            l = np.maximum(point-w, point*0)
            u = point+w
            region = image[tuple([slice(i,j+1) for i,j in zip(l,u)])]
            peak_ind = tuple(np.unravel_index(np.argmax(region), region.shape)+point-np.minimum(point, 0*point+w))
            point = tuple(point)
            if np.all(image[peak_ind] == image[point]): # Can't compare directly in case neighbours have same value
                return point
            return find_local_maximum(image, peak_ind)
        def best_layer(layers):
            for l in layers:
                if l.visible:
                    return l
            return layers[0]
        # Step 2: Processe base layer click
        def base_click_callback(viewer, e):
            if e.type != "mouse_press":
                return
            # If right click, find the nearby peak
            if e.button == 2: # Right click
                pos = find_local_maximum(best_layer(layers_base).data, e.position)
            else:
                pos = e.position
            # Step 2a: Process base layer click
            temp_points.append(pos)
            for layer_base in layers_base:
                layer_base.mouse_drag_callbacks.pop()
            for layer_movable in layers_movable:
                layer_movable.mouse_drag_callbacks.append(movable_click_callback)
            layer_base_points.data = np.vstack([layer_base_points.data, pos])
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
            # If right click, find the nearby peak
            if e.button == 2: # Right click
                bl = best_layer(layers_movable)
                pos = find_local_maximum(bl.data, e.position - bl.translate) + bl.translate
            else:
                pos = e.position
            # Step 3a: Process movable layer click
            base_points.append(temp_points[0])
            movable_points.append(pretransform.transform(utils.invert_transform_numerical(tform, pos)))
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
    def apply_transform(*args, transform=None, force=True, **kwargs):
        # kwargs here are extra parameters to pass to the transform.
        nonlocal tform, movable_points, params, _prev_matrix, _prev_translate
        if transform is not None:
            tform = transform
            if is_point_transform:
                layer_movable_points.data = tform.transform(pretransform.inverse_transform(movable_points))
                layer_movable_points.refresh()
        elif is_point_transform:
            if movable_points is not None and len(movable_points) > 0:
                tform = tform_type(points_start=movable_points, points_end=base_points, **params)
                layer_movable_points.data = tform.transform(pretransform.inverse_transform(movable_points))
            else:
                tform = pretransform
                layer_movable_points.data = np.asarray([])
            layer_movable_points.refresh()
        else:
            tform = tform_type(**params)
        for b in buttons: # Disable buttons while applying transform
            b.enabled = False
        for layer_movable,mi in zip(layers_movable,movable_image):
            # This if statement is a special case optimisation for
            # AffineTransforms only to avoid rerending the image if only the
            # origin/translation has changed.
            if force or _prev_matrix is None or (not isinstance(tform, AffineTransform)) or (isinstance(tform, AffineTransform) and np.any(_prev_matrix != tform.matrix)):
                layer_movable.data = tform.transform_image(mi, relative=rel, labels=utils.image_is_label(mi), downsample=downsample)
                layer_movable.translate = tform.origin_and_maxpos(mi, relative=rel)[0]
            else:
                # This is complicated due to the possibilty of dragging a cropped image out of the crop boundaries
                layer_movable.translate = _prev_translate - tform.shift
            layer_movable.refresh()
        if isinstance(tform, AffineTransform) and (np.any(_prev_matrix != tform.matrix) or force):
            _prev_matrix = tform.matrix
            _prev_translate = tform.origin_and_maxpos(mi, relative=rel)[0] + tform.shift
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
                apply_transform(force=False)
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
