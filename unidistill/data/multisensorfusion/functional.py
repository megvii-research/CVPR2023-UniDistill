import numba
import numpy as np
from PIL import Image


def camera_to_lidar(points, r_rect, velo2cam):
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    xyz = data[:, 0:3]
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def corners_nd(dims, origin=0.5):
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    if ndim == 2:
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2**ndim, ndim])
    return corners


def rotation_2d(points, angles):
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


@numba.jit(nopython=True)
def depth_to_points(depth, trunc_pixel):
    num_pts = np.sum(depth[trunc_pixel:,] > 0.1)
    points = np.zeros((num_pts, 3), dtype=depth.dtype)
    x = np.array([0, 0, 1], dtype=depth.dtype)
    k = 0
    for i in range(trunc_pixel, depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i, j] > 0.1:
                x = np.array([j, i, 1], dtype=depth.dtype)
                points[k] = x * depth[i, j]
                k += 1
    return points


def depth_to_lidar_points(depth, trunc_pixel, P2, r_rect, velo2cam):
    pts = depth_to_points(depth, trunc_pixel)
    points_shape = list(pts.shape[0:-1])
    points = np.concatenate([pts, np.ones(points_shape + [1])], axis=-1)
    points = points @ np.linalg.inv(P2.T)
    lidar_points = camera_to_lidar(points, r_rect, velo2cam)
    return lidar_points


def rotation_3d_in_axis(points, angles, axis=0):
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 0:
        rot_mat_T = np.stack(
            [
                [ones, zeros, zeros],
                [zeros, rot_cos, rot_sin],
                [zeros, -rot_sin, rot_cos],
            ]
        )

    elif axis == 1:
        rot_mat_T = np.stack(
            [
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos],
            ]
        )

    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                [rot_cos, rot_sin, zeros],
                [-rot_sin, rot_cos, zeros],
                [zeros, zeros, ones],
            ]
        )
    else:
        raise ValueError("axis should in range")

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 1.0, 0.5), axis=1):
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes):
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(1, 4, 2)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = rot_sin
        rot_mat_T[1, 0] = -rot_sin
        rot_mat_T[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners


@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array(
        [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7]
    ).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces


def rotation_points_single_angle(points, angle, axis=0):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
            dtype=points.dtype,
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
            dtype=points.dtype,
        )
    elif axis == 0:
        rot_mat_T = np.array(
            [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
            dtype=points.dtype,
        )
    else:
        raise ValueError("axis should in range")

    return points @ rot_mat_T, rot_mat_T


def points_cam2img(points_3d, proj_mat, with_depth=False):
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, (
        "The dimension of the projection"
        f" matrix should be 2 instead of {len(proj_mat.shape)}."
    )
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (d1 == 4 and d2 == 4), (
        "The shape of the projection matrix" f" ({d1}*{d2}) is not supported."
    )
    if d1 == 3:
        proj_mat_expanded = np.eye(4, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    points_4 = np.concatenate([points_3d, np.ones(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        points_2d_depth = np.concatenate([point_2d_res, point_2d[..., 2:3]], axis=-1)
        return points_2d_depth

    return point_2d_res


def box3d_to_bbox(box3d, P2):
    box_corners = center_to_corner_box3d(
        box3d[:, :3], box3d[:, 3:6], box3d[:, 6], [0.5, 1.0, 0.5], axis=1
    )
    box_corners_in_image = points_cam2img(box_corners, P2)
    minxy = np.min(box_corners_in_image, axis=1)
    maxxy = np.max(box_corners_in_image, axis=1)
    bbox = np.concatenate([minxy, maxxy], axis=1)
    return bbox


def corner_to_surfaces_3d(corners):
    surfaces = np.array(
        [
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
            [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
            [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
        ]
    ).transpose([2, 0, 1, 3])
    return surfaces


def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0)):
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def minmax_to_corner_2d(minmax_box):
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box2d(center, dims, origin=0.0)


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


def create_anchors_3d_range(
    feature_size,
    anchor_range,
    sizes=((1.6, 3.9, 1.56),),
    rotations=(0, np.pi / 2),
    dtype=np.float32,
):
    anchor_range = np.array(anchor_range, dtype)
    z_centers = np.linspace(
        anchor_range[2], anchor_range[5], feature_size[0], dtype=dtype
    )
    y_centers = np.linspace(
        anchor_range[1], anchor_range[4], feature_size[1], dtype=dtype
    )
    x_centers = np.linspace(
        anchor_range[0], anchor_range[3], feature_size[2], dtype=dtype
    )
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(x_centers, y_centers, z_centers, rotations, indexing="ij")
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


def center_to_minmax_2d(centers, dims, origin=0.5):
    if origin == 0.5:
        return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])


def rbbox2d_to_near_bbox(rbboxes):
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, mode="iou", eps=0.0):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + eps) * (
            query_boxes[k, 3] - query_boxes[k, 1] + eps
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2])
                - max(boxes[n, 0], query_boxes[k, 0])
                + eps
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3])
                    - max(boxes[n, 1], query_boxes[k, 1])
                    + eps
                )
                if ih > 0:
                    if mode == "iou":
                        ua = (
                            (boxes[n, 2] - boxes[n, 0] + eps)
                            * (boxes[n, 3] - boxes[n, 1] + eps)
                            + box_area
                            - iw * ih
                        )
                    else:
                        ua = (boxes[n, 2] - boxes[n, 0] + eps) * (
                            boxes[n, 3] - boxes[n, 1] + eps
                        )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def projection_matrix_to_CRT_kitti(proj):
    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def remove_outside_points(points, rect, Trv2c, P2, image_shape):
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    points = points[indices.reshape([-1])]
    return points


def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]], dtype=C.dtype
    )
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype
    )
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype
    )
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz


def surface_equ_3d(polygon_surfaces):
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    d = np.einsum("aij, aij->ai", normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


@numba.njit
def _points_in_convex_polygon_3d_jit(
    points, polygon_surfaces, normal_vec, d, num_surfaces
):
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    return _points_in_convex_polygon_3d_jit(
        points, polygon_surfaces, normal_vec, d, num_surfaces
    )


@numba.jit
def points_in_convex_polygon_jit(points, polygon, clockwise=True):
    num_points_of_polygon = polygon.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon.shape[0]
    vec1 = np.zeros((2), dtype=polygon.dtype)
    ret = np.zeros((num_points, num_polygons), dtype=np.bool_)
    success = True
    cross = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            success = True
            for k in range(num_points_of_polygon):
                if clockwise:
                    vec1 = polygon[j, k] - polygon[j, k - 1]
                else:
                    vec1 = polygon[j, k - 1] - polygon[j, k]
                cross = vec1[1] * (polygon[j, k, 0] - points[i, 0])
                cross -= vec1[0] * (polygon[j, k, 1] - points[i, 1])
                if cross >= 0:
                    success = False
                    break
            ret[i, j] = success
    return ret


def boxes3d_to_corners3d_lidar(boxes3d, bottom_center=True):
    boxes_num = boxes3d.shape[0]
    w, l, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array(
        [w / 2.0, -w / 2.0, -w / 2.0, w / 2.0, w / 2.0, -w / 2.0, -w / 2.0, w / 2.0],
        dtype=np.float32,
    ).T
    y_corners = np.array(
        [-l / 2.0, -l / 2.0, l / 2.0, l / 2.0, -l / 2.0, -l / 2.0, l / 2.0, l / 2.0],
        dtype=np.float32,
    ).T
    if bottom_center:
        z_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        z_corners[:, 4:8] = h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        z_corners = np.array(
            [
                -h / 2.0,
                -h / 2.0,
                -h / 2.0,
                -h / 2.0,
                h / 2.0,
                h / 2.0,
                h / 2.0,
                h / 2.0,
            ],
            dtype=np.float32,
        ).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(
        ry.size, dtype=np.float32
    )
    rot_list = np.array(
        [
            [np.cos(ry), -np.sin(ry), zeros],
            [np.sin(ry), np.cos(ry), zeros],
            [zeros, zeros, ones],
        ]
    )
    R_list = np.transpose(rot_list, (2, 0, 1))

    temp_corners = np.concatenate(
        (
            x_corners.reshape(-1, 8, 1),
            y_corners.reshape(-1, 8, 1),
            z_corners.reshape(-1, 8, 1),
        ),
        axis=2,
    )
    rotated_corners = np.matmul(temp_corners, R_list)
    x_corners = rotated_corners[:, :, 0]
    y_corners = rotated_corners[:, :, 1]
    z_corners = rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate(
        (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2
    )

    return corners.astype(np.float32)


def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    large_boxes3d = boxes3d.copy()
    large_boxes3d[:, 3:6] += np.array(extra_width)[None, :]
    return large_boxes3d


def remove_points_in_boxes(points, boxes):
    masks = points_in_rbbox(points, boxes)
    points = points[np.logical_not(masks.any(-1))]
    return points


def boxes3d_kitti_fakelidar_to_lidar(boxes3d_fakelidar):
    w, l, h, r = (
        boxes3d_fakelidar[:, 3:4],
        boxes3d_fakelidar[:, 4:5],
        boxes3d_fakelidar[:, 5:6],
        boxes3d_fakelidar[:, 6:7],
    )
    boxes3d_fakelidar[:, 2] += h[:, 0] / 2
    return np.concatenate(
        [boxes3d_fakelidar[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1
    )


def img_transform(img, resize, resize_dims, crop, flip, rotate):
    ida_rot = np.eye(2)
    ida_tran = np.zeros(2)
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    ida_rot *= resize
    ida_tran -= np.array(crop[:2])
    if flip:
        A = np.array([[-1, 0], [0, 1]])
        b = np.array([crop[2] - crop[0], 0])
        ida_rot = A @ ida_rot
        ida_tran = A @ ida_tran + b
    ang = rotate / 180 * np.pi
    A = np.array(
        [
            [np.cos(ang), np.sin(ang)],
            [-np.sin(ang), np.cos(ang)],
        ]
    )
    b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A @ (-b) + b
    ida_rot = A @ ida_rot
    ida_tran = A @ ida_tran + b
    ida_mat = np.zeros((4, 4))
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat


def bev_transform(gt_boxes, rotate_angle, scale_ratio, trans, flip_dx, flip_dy):
    rotate_angle = rotate_angle / 180 * np.pi
    rot_sin = np.sin(rotate_angle)
    rot_cos = np.cos(rotate_angle)
    rot_mat = np.array(
        [
            [rot_cos, -rot_sin, 0, 0],
            [rot_sin, rot_cos, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    scale_mat = np.array(
        [
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1],
        ]
    )
    trans_mat = np.array(
        [
            [1, 0, 0, trans[0]],
            [0, 1, 0, trans[1]],
            [0, 0, 1, trans[2]],
            [0, 0, 0, 1],
        ]
    )
    flip_mat = np.eye(4)
    if flip_dx:
        flip_mat = flip_mat @ np.array(
            [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
    if flip_dy:
        flip_mat = flip_mat @ np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
    transform_mat = flip_mat @ trans_mat @ scale_mat @ rot_mat
    if gt_boxes.shape[0] > 0:
        homogeneous_center = np.ones((gt_boxes.shape[0], 4))
        homogeneous_center[:, :3] = gt_boxes[:, :3]
        homogeneous_center_transform = (transform_mat @ homogeneous_center.T).T
        gt_boxes[:, :3] = homogeneous_center_transform[:, :3]
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        if flip_dx:
            gt_boxes[:, 6] = np.pi - gt_boxes[:, 6]
        if flip_dy:
            gt_boxes[:, 6] = -gt_boxes[:, 6]
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7:] = (transform_mat[:2, :2] @ gt_boxes[:, 7:].T).T
    return gt_boxes, transform_mat
