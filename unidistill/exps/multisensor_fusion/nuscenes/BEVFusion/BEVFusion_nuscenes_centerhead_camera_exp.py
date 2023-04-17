from unidistill.exps.base_cli import run_cli
from unidistill.exps.multisensor_fusion.nuscenes.BEVFusion.BEVFusion_nuscenes_centerhead_fusion_exp import (
    Exp as BaseExp,
)


class Exp(BaseExp):
    def __init__(
        self,
        batch_size_per_device=4,
        total_devices=1,
        max_epochs=20,
        ckpt_path=None,
        **kwargs
    ):
        super(Exp, self).__init__(
            batch_size_per_device, total_devices, max_epochs, ckpt_path
        )
        self.lr = 2e-4
        self.lr_scale_factor = {"camera_encoder": 1.0}
        self.data_cfg["lidar_key_list"] = []
        self.model_cfg["lidar_encoder"] = None
        self.model = self._configure_model()
        self.train_dataloader = self.configure_train_dataloader()
        self.val_dataloader = self.configure_val_dataloader()
        self.test_dataloader = self.configure_test_dataloader()

    def _change_cfg_params(self):
        self.data_cfg["aug_cfg"]["gt_sampling_cfg"] = None


if __name__ == "__main__":
    import logging

    logging.getLogger("mmcv").disabled = True
    logging.getLogger("mmseg").disabled = True
    run_cli(Exp, "BEVFusion_nuscenes_centerhead_camera_exp")
