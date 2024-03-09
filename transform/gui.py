from .base import Identity, Translate, Transform, PointTransform, AffineTransform
import numpy as np
import napari
import magicgui

# Deprecated, functionality is in alignment_gui
def edit_transform(base_image, movable_image, transform):
    return alignment_gui(base_image, movable_image, transform_type=transform.__class__, initial_movable_points=transform.points_start, initial_base_points=transform.points_end)

def alignment_gui(base_image, movable_image, transform_type=Translate, initial_base_points=None, initial_movable_points=None, references=[]):
    """Align images

    If `base_image` and/or `movable_image` are tuples, they will be interpreted
    as multi-channel

    Reference should be a list of tuples, where each tuple is (image, transform)

    """
    if not isinstance(base_image, tuple):
        base_image = tuple([base_image])
    if not isinstance(movable_image, tuple):
        movable_image = tuple([movable_image])
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
    if is_point_transform:
        assert initial_base_points is None
        assert initial_movable_points is None
    _prev_matrix = None # A special case optimisation for linear transforms
    v = napari.Viewer()
    # v.window._qt_viewer._dockLayerList.setVisible(False)
    # v.window._qt_viewer._dockLayerControls.setVisible(False)
    base_points = [] if initial_base_points is None else list(initial_base_points)
    movable_points = [] if initial_movable_points is None else list(initial_movable_points)
    tform = Identity()
    tform_type = transform_type
    layers_base = [v.add_image(bi, colormap="red", blending="additive", name="base") for bi in base_image]
    layers_movable = [v.add_image(tform.transform_image(mi, relative=True), colormap="blue", blending="additive", name="movable", translate=tform.origin) for mi in movable_image]
    layers_reference = [v.add_image(rt.transform_image(ri, relative=True), colormap="green", blending="additive", name=f"reference_{i}", translate=rt.origin) for i,(ri,rt) in references]
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
            print("Base layer callback")
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
            print("Movable layer callback")
            # Step 3a: Process movable layer click
            base_points.append(temp_points[0])
            movable_points.append(tform.inverse_transform([e.position])[0])
            for layer_movable in layers_movable:
                layer_movable.mouse_drag_callbacks.pop()
            for layer_base in layers_base:
                layer_base.opacity = 1
            # Step 3b: Clean up after clicks
            layer_base_points.data = base_points
            layer_movable_points.data = tform.transform(movable_points)
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
            print("Base layer callback")
            v.mouse_drag_callbacks.pop()
            # Step 2a: Find and remove the closest point (base or movable) to the click and its corresponding point (movable or base)
            search_point = e.position
            dists_base = np.sum(np.square(np.asarray(base_points) - [search_point]), axis=1)
            dists_movable = np.sum(np.square(np.asarray(tform.transform(movable_points)) - [search_point]), axis=1)
            ind = np.argmin(dists_base) if np.min(dists_base) < np.min(dists_movable) else np.argmin(dists_movable)
            base_points.pop(ind)
            movable_points.pop(ind)
            # Step 2b: Clean up
            for layer_base in layers_base:
                layer_base_points.data = base_points
            for layer_movable in layers_movable:
                layer_movable_points.data = tform.transform(movable_points)
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
    def apply_transform(*args, transform_type=None, **kwargs):
        # kwargs here are extra parameters to pass to the transform.
        nonlocal tform, movable_points, params, _prev_matrix
        print(params)
        if transform_type is None:
            transform_type = tform_type
        if is_point_transform:
            if movable_points is None or len(movable_points) == 0:
                transform_type = Identity
            tform = transform_type(points_start=movable_points, points_end=base_points, input_bounds=movable_image[0].shape, **params)
            layer_movable_points.data = tform.transform(movable_points)
            layer_movable_points.refresh()
        else:
            tform = transform_type(input_bounds=movable_image[0].shape, **params)
        for b in buttons: # Disable buttons while applying transform
            b.enabled = False
        for layer_movable,mi in zip(layers_movable,movable_image):
            # This if statement is a special case optimisation for
            # AffineTransforms only to avoid rerending the image if only the
            # origin/translation has changed.
            if _prev_matrix is None or (isinstance(tform, AffineTransform) and np.any(_prev_matrix != tform.matrix)):
                layer_movable.data = tform.transform_image(mi, relative=True)
            if isinstance(tform, AffineTransform):
                _prev_matrix = tform.matrix
            layer_movable.translate = tform.origin
            layer_movable.refresh()
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
    button_reset.clicked.connect(lambda : apply_transform(transform_type=Identity))
    button_delete = magicgui.widgets.PushButton(value=True, text='Remove point')
    button_delete.clicked.connect(remove_point)
    if is_point_transform:
        buttons = [button_add_point, button_transform, button_reset, button_delete]
    else:
        buttons = [button_transform, button_reset]
    widgets = []
    widgets.extend(buttons)
    for p,pv in params.items():
        # This currently assumes all parameters are floats
        spinbox = magicgui.widgets.FloatSpinBox(value=pv, label=p+":")
        def spinbox_callback(*args,p=p,spinbox=spinbox):
            params[p] = spinbox.value
            if dynamic_update.value:
                apply_transform()
            print(params, "From callback")
        spinbox.changed.connect(spinbox_callback)
        widgets.append(spinbox)
    dynamic_update = magicgui.widgets.CheckBox(value=False, label="Dynamic update")
    if len(params) > 0:
        widgets.append(dynamic_update)
    container_widget = magicgui.widgets.Container(widgets=widgets)
    v.window.add_dock_widget(container_widget, area="left", add_vertical_stretch=False)
    if is_point_transform:
        v.camera.events.zoom.connect(set_point_size)
        set_point_size()
    apply_transform()
    v.show(block=True)
    return tform
