import tensorflow as tf # 2.9.0
import numpy as np # 1.22.4

# print(tf.__version__)
# print(np.__version__)

def optical_flow_sum(mesh, optical_flow_chain, indexes, flow, mask):
    h, w = flow.shape[:2]
    # mask the correct points (the ones that are inside image and on mask of frame t-1)
    # Check (in x and y axis) which points from previous frame t-1 are present in current frame t
    xx = np.logical_and((0 <= optical_flow_chain[:, 0]), (optical_flow_chain[:, 0] <= w-1))
    yy = np.logical_and((0 <= optical_flow_chain[:, 1]), (optical_flow_chain[:, 1] <= h-1))
    # And limit the optical flow chain to only this points from an actual frame t
    optical_flow_chain = optical_flow_chain[np.logical_and(xx, yy)]

    # Same for the mask for frame t
    mm = mask[optical_flow_chain[:, 1].astype(np.int32), optical_flow_chain[:, 0].astype(np.int32)]
    # Use mm to retrieve mask points from optical flow chain
    optical_flow_chain = optical_flow_chain[mm]

    if mesh is not None:
        # Limit the mesh in the same way like optical flow chain - to the mask points from frame t-1
        mesh = mesh[np.logical_and(xx, yy)]
        mesh = mesh[mm]

    # Same for indexes
    indexes = indexes[np.logical_and(xx, yy)]
    indexes = indexes[mm]

    # bilinear interpolation
    optical_flow_chain_start = optical_flow_chain.astype(int)
    # Retrieve points which have floating point coordinates
    fractions = optical_flow_chain - optical_flow_chain_start

    q00 = flow[optical_flow_chain_start[:, 1],   optical_flow_chain_start[:, 0]]                                  # upper left
    q01 = flow[np.clip(optical_flow_chain_start[:, 1]+1,0,h-1), optical_flow_chain_start[:, 0]]                   # lower left
    q10 = flow[optical_flow_chain_start[:, 1],   np.clip(optical_flow_chain_start[:, 0]+1,0,w-1)]                 # upper right
    q11 = flow[np.clip(optical_flow_chain_start[:, 1]+1,0,h-1), np.clip(optical_flow_chain_start[:, 0]+1,0,w-1)]  # lower right
    one_minus_fraction = (1-fractions)

    _1_frac_x = one_minus_fraction[:,0]
    _1_frac_y = one_minus_fraction[:,1]
    frac_x = fractions[:,0]
    frac_y = fractions[:,1]

    W00 = _1_frac_x * _1_frac_y
    W01 = frac_y * _1_frac_x
    W10 = frac_x * _1_frac_y
    W11 = frac_x * frac_y

    optical_flow_chain += np.multiply(q00, np.dstack((W00,W00))[0]) +\
                        np.multiply(q01, np.dstack((W01, W01))[0]) + \
                        np.multiply(q10, np.dstack((W10, W10))[0]) + \
                        np.multiply(q11, np.dstack((W11, W11))[0])
    # mesh is an array (no_points, 2) of coordinates of points of frame t-1
    # optical_flow_chain is an array (no_points, 2) of coordinates of mask points from frame t-1 in frame t
    return mesh, optical_flow_chain, indexes

def JEPE(of12, of23, of13):
    metric = {}
    of12 = of12[0]
    of23 = of23[0]
    of13 = of13[0]
    w_f, h_f, c = of12.shape # 800 800 2

    x = np.arange(0, w_f, 1) # [0 1 2 ... 799]
    y = np.arange(0, h_f, 1)
    xx, yy = np.meshgrid(x, y)
    mesh = np.dstack((xx, yy)).reshape(-1, 2).astype(np.float64) # points coordinates - [[0. 0.], [1. 0.] ... [799. 799.]
    optical_flow_chain = mesh.copy()
    indexes = np.arange(mesh.shape[0]) # [0, 1 ... 639999]

    # mask ones when all pixels of OF are adding
    mask = np.ones((w_f, h_f))
    mask = (mask > 0)

    _optical_flow_chain = np.zeros_like(optical_flow_chain)
    _optical_flow_chain_13 = np.zeros_like(optical_flow_chain)

    mesh_13, optical_flow_chain_13, indexes_13 = optical_flow_sum(mesh, optical_flow_chain, indexes, of13, mask)
    mesh, optical_flow_chain, indexes = optical_flow_sum(mesh, optical_flow_chain, indexes, of12, mask)
    mesh, optical_flow_chain, indexes = optical_flow_sum(mesh, optical_flow_chain, indexes, of23, mask)

    filtered_x = np.take((optical_flow_chain_13 - mesh_13)[:, 0], indexes)
    filtered_y = np.take((optical_flow_chain_13 - mesh_13)[:, 1], indexes)
    optical_flow_chain = optical_flow_chain-mesh

    metric['JEPE'] = np.mean(np.sqrt((optical_flow_chain[:,0]-filtered_x)**2 + (optical_flow_chain[:,1]-filtered_y)**2 ))
    return metric

def JEPE3(of12, of23, of34, of14):
    metric = {}
    of12 = of12[0]
    of23 = of23[0]
    of34 = of34[0]
    of14 = of14[0]
    w_f, h_f, c = of12.shape # 800 800 2

    x = np.arange(0, w_f, 1) # [0 1 2 ... 799]
    y = np.arange(0, h_f, 1)
    xx, yy = np.meshgrid(x, y)
    mesh = np.dstack((xx, yy)).reshape(-1, 2).astype(np.float64) # points coordinates - [[0. 0.], [1. 0.] ... [799. 799.]
    optical_flow_chain = mesh.copy()
    indexes = np.arange(mesh.shape[0]) # [0, 1 ... 639999]

    # mask ones when all pixels of OF are adding
    mask = np.ones((w_f, h_f))
    mask = (mask > 0)

    _optical_flow_chain = np.zeros_like(optical_flow_chain)
    _optical_flow_chain_13 = np.zeros_like(optical_flow_chain)

    mesh_14, optical_flow_chain_14, indexes_14 = optical_flow_sum(mesh, optical_flow_chain, indexes, of14, mask)
    mesh, optical_flow_chain, indexes = optical_flow_sum(mesh, optical_flow_chain, indexes, of12, mask)
    mesh, optical_flow_chain, indexes = optical_flow_sum(mesh, optical_flow_chain, indexes, of23, mask)
    mesh, optical_flow_chain, indexes = optical_flow_sum(mesh, optical_flow_chain, indexes, of34, mask)

    filtered_x = np.take((optical_flow_chain_14 - mesh_14)[:, 0], indexes)
    filtered_y = np.take((optical_flow_chain_14 - mesh_14)[:, 1], indexes)
    optical_flow_chain = optical_flow_chain-mesh

    metric['JEPE3'] = np.mean(np.sqrt((optical_flow_chain[:,0]-filtered_x)**2 + (optical_flow_chain[:,1]-filtered_y)**2 ))
    return metric


def image_warp(im, flow):
    """Performs a backward warp of an image using the predicted flow.

    Args:
        im: Batch of images. [num_batch, height, width, channels]
        flow: Batch of flow vectors. [num_batch, height, width, 2]
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    with tf.compat.v1.variable_scope('image_warp'):
        num_batch, height, width, channels = tf.unstack(tf.shape(im))
        max_x = tf.cast(width - 1, 'int32')
        max_y = tf.cast(height - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # We have to flatten our tensors to vectorize the interpolation
        im_flat = tf.reshape(im, [-1, channels])
        flow_flat = tf.reshape(flow, [-1, 2])

        # Floor the flow, as the final indices are integers
        # The fractional part is used to control the bilinear interpolation.

        flow_floor = tf.cast(tf.floor(flow_flat), tf.int32)
        bilinear_weights = flow_flat - tf.floor(flow_flat)

        # Construct base indices which are displaced with the flow
        pos_x = tf.tile(tf.range(width), [height * num_batch])
        grid_y = tf.tile(tf.expand_dims(tf.range(height), 1), [1, width])
        pos_y = tf.tile(tf.reshape(grid_y, [-1]), [num_batch])

        x = flow_floor[:, 0]
        y = flow_floor[:, 1]
        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute interpolation weights for 4 adjacent pixels
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa = tf.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
        wb = tf.expand_dims((1 - xw) * yw, 1) # bottom left pixel
        wc = tf.expand_dims(xw * (1 - yw), 1) # top right pixel
        wd = tf.expand_dims(xw * yw, 1) # bottom right pixel

        x0 = pos_x + x
        x1 = x0 + 1
        y0 = pos_y + y
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim1 = width * height
        batch_offsets = tf.range(num_batch) * dim1
        base_grid = tf.tile(tf.expand_dims(batch_offsets, 1), [1, dim1])
        base = tf.reshape(base_grid, [-1])

        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        warped_flat = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        warped = tf.reshape(warped_flat, [num_batch, height, width, channels])

        return warped

def create_mask(tensor, paddings):
    with tf.compat.v1.variable_scope('create_mask'):
        shape = tf.shape(tensor)
        inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
        inner = tf.ones([inner_width, inner_height])

        mask2d = tf.pad(inner, paddings)
        mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
        mask4d = tf.expand_dims(mask3d, 3)
        return tf.stop_gradient(mask4d)
def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

def compute_metrics(flow_fw, flow_bw):
    metrics = {}
    flow_bw_warped = image_warp(flow_bw, flow_fw)
    flow_fw_warped = image_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped

    metrics['Ec'] = (charbonnier_loss(flow_diff_fw) + charbonnier_loss(flow_diff_bw)).numpy()  # Ec
    metrics['smooth_2nd'] = (second_order_loss(flow_fw) + second_order_loss(flow_bw)).numpy()  # Es
    return metrics

def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.

    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    with tf.compat.v1.variable_scope('charbonnier_loss'):
        batch, height, width, channels = tf.unstack(tf.shape(x))
        normalization = tf.cast(batch * height * width * channels, tf.float32)
        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)
        if mask is not None:
            error = tf.multiply(mask, error)
        if truncate is not None:
            error = tf.minimum(error, truncate)
        return tf.reduce_sum(error) / normalization

def _second_order_deltas(flow):
    with tf.compat.v1.variable_scope('_second_order_deltas'):
        mask_x = create_mask(flow, [[0, 0], [1, 1]])
        mask_y = create_mask(flow, [[1, 1], [0, 0]])
        mask_diag = create_mask(flow, [[1, 1], [1, 1]])
        mask = tf.concat(axis=3, values=[mask_x, mask_y, mask_diag, mask_diag])

        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2
        weights = tf.constant(weight_array, dtype=tf.float32)

        flow_u, flow_v = tf.split(axis=3, num_or_size_splits=2, value=flow)
        delta_u = conv2d(flow_u, weights)
        delta_v = conv2d(flow_v, weights)
        return delta_u, delta_v, mask

def second_order_loss(flow):
    with tf.compat.v1.variable_scope('second_order_loss'):
        delta_u, delta_v, mask = _second_order_deltas(flow)
        loss_u = charbonnier_loss(delta_u, mask)
        loss_v = charbonnier_loss(delta_v, mask)
        return loss_u + loss_v

def t_metrics(OF12, OF23, OF13, OF21, OF34=None, OF14=None):
    metrics = {}
    if OF21 is not None:
        metrics.update(compute_metrics(OF12, OF21))   # compute Ec and Smoothness between forward and backward flow
    if OF13 is not None:
        metrics.update(JEPE(OF12, OF23, OF13)) # compute JEPE
    if OF34 is not None and OF14 is not None:
        metrics.update(JEPE3(OF12, OF23, OF34, OF14))  # compute JEPE3
    return metrics

if __name__ == '__main__':
    # Random flow
    OF12 = np.random.rand(1, 800, 800, 2).astype(np.float32) # forward flow between frame 1 and 2
    OF23 = np.random.rand(1, 800, 800, 2).astype(np.float32) # forward flow between frame 2 and 3
    OF13 = np.random.rand(1, 800, 800, 2).astype(np.float32) # forward flow between frame 1 and 3
    OF21 = np.random.rand(1, 800, 800, 2).astype(np.float32) # backward flow between frame 1 and 2

    # Compute metrics
    metrics = t_metrics(OF12, OF23, OF13, OF21)
    print(metrics)


    # # Test JEPE3
    # OF12 = np.random.rand(1, 800, 800, 2).astype(np.float32) # forward flow between frame 1 and 2
    # OF23 = np.random.rand(1, 800, 800, 2).astype(np.float32) # forward flow between frame 2 and 3
    # OF34 = np.random.rand(1, 800, 800, 2).astype(np.float32) # forward flow between frame 3 and 4
    # OF13 = np.random.rand(1, 800, 800, 2).astype(np.float32) # forward flow between frame 1 and 3
    # OF14 = np.random.rand(1, 800, 800, 2).astype(np.float32) # forward flow between frame 1 and 4
    # OF21 = np.random.rand(1, 800, 800, 2).astype(np.float32) # backward flow between frame 1 and 2
    #
    # metrics = t_metrics(OF12, OF23, OF13, OF21, OF34, OF14)
    # print(metrics)
    #
    # if metrics['JEPE'] <= 1.0 and metrics['Ec'] <= 1.0:
    #     print('True')
