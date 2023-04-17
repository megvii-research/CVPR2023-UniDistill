_POINT_CLOUD_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
_VOXEL_SIZE = [0.075, 0.075, 0.2]
_GRID_SIZE = [1440, 1440, 40]
_IMG_DIM = (256, 704)
_OUT_SIZE_FACTOR = 8

COMMON_CFG = dict(
    point_cloud_range=_POINT_CLOUD_RANGE,
    voxel_size=_VOXEL_SIZE,
    grid_size=_GRID_SIZE,
    img_dim=_IMG_DIM,
    out_size_factor=_OUT_SIZE_FACTOR,
)

CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

_AUG_CFG = dict(
    point_cloud_range=_POINT_CLOUD_RANGE,
    img_norm_cfg=dict(
        img_mean=[123.675, 116.28, 103.53], img_std=[58.395, 57.12, 57.375], to_rgb=True
    ),
    ida_aug_cfg=dict(
        resize_lim=(0.386, 0.55),
        final_dim=_IMG_DIM,
        rot_lim=(-5.4, 5.4),
        H=900,
        W=1600,
        rand_flip=True,
        bot_pct_lim=(0.0, 0.0),
    ),
    bda_aug_cfg=dict(
        rot_lim=(-22.5 * 2, 22.5 * 2),
        scale_lim=(0.90, 1.10),
        trans_lim=(0.5, 0.5, 0.5),
        flip_dx_ratio=0.5,
        flip_dy_ratio=0.5,
    ),
    gt_sampling_cfg=dict(
        root_path="/data/dataset",
        data_name="nuScenes_multimodal",  # optional: nuScenes
        data_split="training",
        use_road_plane=False,
        stop_epoch=16,  # Not supported now
        filter_by_min_points_cfg=[
            "car:5",
            "truck:5",
            "construction_vehicle:5",
            "bus:5",
            "trailer:5",
            "barrier:5",
            "motorcycle:5",
            "bicycle:5",
            "pedestrian:5",
            "traffic_cone:5",
        ],
        num_point_feature=5,
        remove_extra_width=[0.0, 0.0, 0.0],
        limit_whole_scene=True,
        sampler_groups=[
            "car:2",
            "truck:3",
            "construction_vehicle:7",
            "bus:4",
            "trailer:6",
            "barrier:2",
            "motorcycle:6",
            "bicycle:6",
            "pedestrian:2",
            "traffic_cone:2",
        ],
        class_names=CLASS_NAMES,
    ),
)

DATA_CFG = dict(
    root_path="/data/dataset",
    lidar_key_list=["LIDAR_TOP"],
    img_key_list=[
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
    ],
    num_lidar_sweeps=10,
    num_cam_sweeps=0,
    lidar_with_timestamp=True,
    class_names=CLASS_NAMES,
    use_cbgs=True,
    aug_cfg=_AUG_CFG,
)

MODEL_CFG = dict(
    class_names=CLASS_NAMES,
    lidar_encoder=dict(
        point_cloud_range=_POINT_CLOUD_RANGE,
        voxel_size=_VOXEL_SIZE,
        grid_size=_GRID_SIZE,
        max_num_points=10,
        max_voxels=(120000, 160000),
        src_num_point_features=5,
        use_num_point_features=5,
        map_to_bev_num_features=256,
    ),
    camera_encoder=dict(
        x_bound=[
            _POINT_CLOUD_RANGE[0],
            _POINT_CLOUD_RANGE[3],
            _VOXEL_SIZE[0] * _OUT_SIZE_FACTOR,
        ],
        y_bound=[
            _POINT_CLOUD_RANGE[1],
            _POINT_CLOUD_RANGE[4],
            _VOXEL_SIZE[1] * _OUT_SIZE_FACTOR,
        ],
        z_bound=[
            _POINT_CLOUD_RANGE[2],
            _POINT_CLOUD_RANGE[5],
            _POINT_CLOUD_RANGE[5] - _POINT_CLOUD_RANGE[2],
        ],
        d_bound=[2.0, 58.0, 0.5],
        final_dim=_IMG_DIM,
        output_channels=256,
        downsample_factor=16,
        img_backbone_conf=dict(
            type="SwinTransformer",
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=[1, 2, 3],
            with_cp=False,
            convert_weights=True,
            init_cfg=dict(
                type="Pretrained",
                checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
            ),
        ),
        img_neck_conf=dict(
            type="SECONDFPN",
            in_channels=[192, 384, 768],
            upsample_strides=[0.5, 1, 2],
            out_channels=[128, 128, 128],
        ),
        depth_net_conf=dict(in_channels=384, mid_channels=384),
    ),
    bev_encoder=dict(
        backbone2d_layer_nums=[5, 5],
        backbone2d_layer_strides=[1, 2],
        backbone2d_num_filters=[128, 256],
        backbone2d_upsample_strides=[1, 2],
        backbone2d_num_upsample_filters=[256, 256],
        num_bev_features=256,  # sp conv output channel
        backbone2d_use_scconv=False,
    ),
    det_head=dict(
        target_assigner=dict(
            point_cloud_range=_POINT_CLOUD_RANGE,
            voxel_size=_VOXEL_SIZE,
            grid_size=_GRID_SIZE,
            gaussian_overlap=0.1,
            min_radius=2,
            iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
            cls_cost=dict(type="FocalLossCost", gamma=2, alpha=0.25, weight=0.15),
            reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
            iou_cost=dict(type="IoU3DCost", weight=0.25),
        ),
        bbox_coder=dict(
            pc_range=_POINT_CLOUD_RANGE[0:2],
            voxel_size=_VOXEL_SIZE[0:2],
            out_size_factor=_OUT_SIZE_FACTOR,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        dataset_name="nuScenes",
        num_proposals=200,
        hidden_channel=128,
        in_channels=512,
        num_classes=len(CLASS_NAMES),
        num_decoder_layers=1,
        num_heads=8,
        nms_kernel_size=3,
        out_size_factor=_OUT_SIZE_FACTOR,
        common_heads=dict(
            center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
    ),
)


_DENSE_TASKS = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]


CENTERPOINT_DET_HEAD_CFG = dict(
    class_name=CLASS_NAMES,
    target_assigner=dict(
        densehead_out_size_factor=_OUT_SIZE_FACTOR,
        densehead_tasks=_DENSE_TASKS,
        target_assigner_dense_reg=1,
        target_assigner_gaussian_overlap=0.1,
        target_assigner_max_objs=2500,
        target_assigner_min_radius=2,
        target_assigner_mapping={name: idx + 1 for idx, name in enumerate(CLASS_NAMES)},
        grid_size=_GRID_SIZE,
        pc_range=_POINT_CLOUD_RANGE[0:2],
        voxel_size=_VOXEL_SIZE[0:2],
        target_assigner_topk=9,
        target_assigner_no_log=False,
        with_velocity=True,
    ),
    proposal_layer=dict(
        densehead_dataset_name="nuscenes",
        densehead_tasks=_DENSE_TASKS,
        proposal_post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        proposal_score_threshold=0.1,
        proposal_pc_range=_POINT_CLOUD_RANGE[0:2],
        densehead_out_size_factor=_OUT_SIZE_FACTOR,
        proposal_voxel_size=_VOXEL_SIZE[0:2],
        no_log=False,
        proposal_iou_aware_list=[0.65] * 10,
        nms_iou_threshold_train=0.8,
        nms_pre_max_size_train=1500,
        nms_post_max_size_train=80,
        nms_iou_threshold_test=0.1,
        nms_pre_max_size_test=1500,
        nms_post_max_size_test=100,
    ),
    dense_head=dict(
        densehead_dataset_name="nuscenes",
        densehead_tasks=_DENSE_TASKS,
        densehead_out_size_factor=_OUT_SIZE_FACTOR,
        input_channels=512,  # need to be careful!
        grid_size=_GRID_SIZE,
        point_cloud_range=_POINT_CLOUD_RANGE,
        densehead_loss_code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        densehead_loss_loc_weight=0.25,
        densehead_loss_iou_weight=5.0,
        densehead_share_conv_channel=64,
        densehead_common_heads=dict(  # common_heads,
            {
                "iou": [1, 2],
                "reg": [2, 2],
                "height": [1, 2],
                "dim": [3, 2],
                "rot": [2, 2],
                "vel": [2, 2],
            }
        ),
        densehead_upsample_for_pedestrian=False,
        densehead_mode="3d",
        densehead_init_bias=-2.19,
    ),
    target_assigner_alpha=0.25,
    target_assigner_gamma=2,
)
