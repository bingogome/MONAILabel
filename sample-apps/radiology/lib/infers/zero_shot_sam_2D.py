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
from typing import Callable, Sequence, Union

from monai.apps.deepedit.transforms import (
    AddGuidanceFromPointsDeepEditd,
    AddGuidanceSignalDeepEditd,
    DiscardAddGuidanced,
    ResizeGuidanceMultipleLabelDeepEditd,
)
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Resized,
    SqueezeDimd,
    ToNumpyd,
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored
from lib.transforms.transforms import SAMTransform, ToCheck
from lib.segment_anything import sam_model_registry


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

    def pre_transforms(self, data=None):
        t = [
            LoadImaged(keys="image", reader="ITKReader"),
            EnsureChannelFirstd(keys="image"),
            SAMTransform(keys="image", sam_model=self.sam_model, device=data.get("device") if data else None),
            ToCheck(keys="image"),
        ]

        self.add_cache_transform(t, data)

        if self.type == InferType.DEEPEDIT:
            t.extend(
                [
                    AddGuidanceFromPointsDeepEditd(ref_image="image", guidance="guidance", label_names=self.labels),
                    Resized(keys="image", spatial_size=self.spatial_size, mode="area"),
                    ResizeGuidanceMultipleLabelDeepEditd(guidance="guidance", ref_image="image"),
                    AddGuidanceSignalDeepEditd(
                        keys="image", guidance="guidance", number_intensity_ch=self.number_intensity_ch
                    ),
                ]
            )
        else:
            t.extend(
                [
                    Resized(keys="image", spatial_size=self.spatial_size, mode="area"),
                    DiscardAddGuidanced(
                        keys="image", label_names=self.labels, number_intensity_ch=self.number_intensity_ch
                    ),
                ]
            )

        t.append(EnsureTyped(keys="image", device=data.get("device") if data else None))
        return t

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
