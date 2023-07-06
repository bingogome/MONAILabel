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
import os
from typing import Callable, Sequence, Dict, Any
import logging
import copy

from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureTyped,
    SqueezeDimd,
    ToNumpyd, LoadImaged, Rotate90d,
)

import torch
import numpy as np
from monai.data import decollate_batch
from monailabel.interfaces.utils.transform import run_transforms
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored
from lib.transforms.transforms import SAMTransform, ToCheck, LoadEmbeddings, ConvertToVolume, ReadPrompts
from lib.segment_anything import sam_model_registry, SamPredictor

# from lib.segment_anything import SamPredictor

logger = logging.getLogger(__name__)


class ZeroShotSam2D(BasicInferTask):
    """
    This provides a sample implementation of SAM pre-trained model.
    """

    def __init__(
            self,
            path,
            network=None,
            type=InferType.DEEPEDIT,
            labels=None,
            dimension=3,
            spatial_size=(128, 128, 64),
            target_spacing=(1.0, 1.0, 1.0),
            number_intensity_ch=1,
            description="SAM model for 2D segmentation",
            **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="image",
            output_label_key="pred",
            output_json_key="result",
            **kwargs,
        )

        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.number_intensity_ch = number_intensity_ch
        self.model_type = 'vit_b'
        self.checkpoint = os.path.split(path[0])[0] + '/sam/sam_vit_b_01ec64.pth'
        self.sam_model = sam_model_registry[self.model_type](checkpoint=self.checkpoint).to('cuda:0')

    def is_valid(self) -> bool:
        return True

    def _get_network(self, device, data):
        path = self.get_path()
        logger.info(f"Infer model path: {path}")
        return self.sam_model

    def run_inferer(self, data: Dict[str, Any], convert_to_batch=True, device="cuda"):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :return: updated data with output_key stored that will be used for post-processing
        """

        inferer = self.inferer(data)
        logger.info(f"Inferer:: {device} => {inferer.__class__.__name__} => {inferer.__dict__}")

        sam_model = self._get_network(device, data)
        predictor = SamPredictor(sam_model)

        # TODO: replace hardcode
        input_point = np.array([[200, 100]])
        input_label = np.array([1])
        input_size = (256,256)
        original_size = (256,256)

        # ``is_image_set'', ``input_size'', ``original_size'' have to be overriden to directly use embedings instead of images
        predictor.is_image_set = True
        predictor.input_size = input_size
        predictor.original_size = original_size
        img_embeddings = torch.as_tensor(data['img_embeddings_axial']).to(device)

        # SAM uses ``features'' attribute, the same as embeddings
        predictor.features = img_embeddings
        masks, scores, logits = self.predictor.predict( \
            point_coords=input_point, \
            point_labels=input_label, \
            multimask_output=True )
        
        data['pred'] = masks[0]

        return data

    def pre_transforms(self, data=None):
        t = [
            LoadImaged(keys='image'),
            LoadEmbeddings(keys="embeddings"),
        ]

        self.add_cache_transform(t, data,
                                 keys=("image", "img_embeddings_axial", "img_embeddings_sagittal", "img_embeddings_coronal"))

        # This is when a prompt (click) is provided
        if self.type == InferType.DEEPEDIT:
            t.extend(
                [
                    ReadPrompts(keys="image", label_names=self.labels),
                ]
            )
        # # This is when NO prompts are provided
        # else:
        #     t.extend(
        #         [
        #             ToCheck(keys="image"),
        #         ]
        #     )
        # t.append(EnsureTyped(keys="image", device=data.get("device") if data else None))
        return t

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    # def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
    #     return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            # EnsureTyped(keys="pred", device=data.get("device") if data else None),
            ConvertToVolume(keys='pred'),
            # Activationsd(keys="pred", sigmoid=True),
            # AsDiscreted(keys="pred", argmax=True),
            # SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
            Rotate90d(keys="pred"),
            Restored(keys="pred", ref_image="image"),
            ToCheck(keys="image"),
        ]
