# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import UNETR, DynUNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)


class ZeroShotSam2D(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        self.epistemic_enabled = None
        self.epistemic_samples = None

        # Multilabel
        self.labels = {
            "labelSAM": 1,
            "background": 0,
        }

        # Number of input channels - 4 for BRATS and 1 for spleen
        self.number_intensity_ch = 1

        network = self.conf.get("network", "dynunet")

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{self.name}_{network}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{self.name}_{network}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "false")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            url = f"{url}/radiology_{network}_multilabel.pt"
            download_file(url, self.path[0])

        self.target_spacing = (1.0, 1.0, 1.0)  # target space for image
        self.spatial_size = (256, 256, 256)  # train input size

        # Network
        self.network = None

        # self.network = (
        #     UNETR(
        #         spatial_dims=3,
        #         in_channels=len(self.labels) + self.number_intensity_ch,
        #         out_channels=len(self.labels),
        #         img_size=self.spatial_size,
        #         feature_size=64,
        #         hidden_size=1536,
        #         mlp_dim=3072,
        #         num_heads=48,
        #         pos_embed="conv",
        #         norm_name="instance",
        #         res_block=True,
        #     )
        #     if network == "unetr"
        #     else DynUNet(
        #         spatial_dims=3,
        #         in_channels=len(self.labels) + self.number_intensity_ch,
        #         out_channels=len(self.labels),
        #         kernel_size=[3, 3, 3, 3, 3, 3],
        #         strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        #         upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        #         norm_name="instance",
        #         deep_supervision=False,
        #         res_block=True,
        #     )
        # )

        # Others
        self.epistemic_enabled = strtobool(conf.get("epistemic_enabled", "false"))
        self.epistemic_samples = int(conf.get("epistemic_samples", "5"))
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_samples}")

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        return {
            self.name: lib.infers.ZeroShotSam2D(
                path=self.path,
                network=self.network,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
                config={"cache_transforms": True, "cache_transforms_in_memory": True, "cache_transforms_ttl": 300},
            ),
            f"{self.name}_seg": lib.infers.ZeroShotSam2D(
                path=self.path,
                network=self.network,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
                number_intensity_ch=self.number_intensity_ch,
                type=InferType.SEGMENTATION,
            ),
        }

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, f"{self.name}_" + self.conf.get("network", "dynunet"))
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.ZeroShotSam2D(
            model_dir=output_dir,
            network=self.network,
            load_path=load_path,
            publish_path=self.path[1],
            spatial_size=self.spatial_size,
            target_spacing=self.target_spacing,
            number_intensity_ch=self.number_intensity_ch,
            config={"pretrained": strtobool(self.conf.get("use_pretrained_model", "false"))},
            labels=self.labels,
            debug_mode=False,
            find_unused_parameters=True,
        )
        return task

