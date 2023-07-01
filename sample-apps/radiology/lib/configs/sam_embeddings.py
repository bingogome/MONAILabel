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
import logging
import numpy as np
import torch
# import SimpleITK as sitk
from scipy import ndimage
from skimage import transform, io, segmentation
from lib.segment_anything import sam_model_registry
from lib.segment_anything.utils.transforms import ResizeLongestSide

from monai.transforms import LoadImage

logger = logging.getLogger(__name__)


class SAMEmbeddings:
    def __init__(self, checkpoint, image_size, model_type):
        self.device = 'cuda:0'
        self.image_size = image_size
        self.lower_bound = -500
        self.upper_bound = 1000
        self.model_type = model_type
        self.checkpoint = checkpoint
        # set up the SAM model
        self.sam_model = sam_model_registry[self.model_type](checkpoint=os.path.join(self.checkpoint, f'sam_{self.model_type}_01ec64.pth')).to(self.device)
        if self.sam_model is None:
            logger.error('SAM pretrained model must be provided')

    def run(self, datastore, label_id):
        # Create npz folder
        embed_path = os.path.join(datastore._datastore_path, 'embeddings')
        os.makedirs(embed_path, exist_ok=True)
        if len(datastore.list_images()) == 0:
            logger.warning("Currently no images are available in datastore for SAM embeddings")
            return
        # List of all labeled images from the datastore
        all_imgs = datastore.list_images()
        loader = LoadImage()
        if len(all_imgs) > 0:
            for j, name in enumerate(all_imgs):
                logger.info(f"Computing embeddings for image {name} - {j}/{len(all_imgs)} ")
                # Reading image
                # image_sitk = sitk.ReadImage(datastore.get_image_uri(name))
                # image_data = sitk.GetArrayFromImage(image_sitk)
                image_data, _ = loader(datastore.get_image_uri(name))
                image_data = image_data.array
                # Reading label/GT
                path_to_gt = datastore.get_label_uri(name, 'final')

                if path_to_gt == '':
                    gt_data = None
                    image_data_pre, axial_index, cor_index, sag_index = self._getIndexes(self, image_data, gt_data)
                    for view, indexes in zip(['axial', 'sagittal', 'coronal'], [axial_index, sag_index, cor_index]):
                        # checking if image already exists
                        if os.path.exists(os.path.join(embed_path, name + f'_{view}.npz')):
                            logger.info(
                                f'SAM embeddings already computed for volume {name} and {view} view')
                            continue
                        else:
                            idx_min, idx_max = np.min(indexes), np.max(indexes)
                            imgs, _, img_embeddings = self._preprocess_img(self, image_data_pre, idx_min, idx_max, view, gt_data)
                            self._save_npz(imgs, [], img_embeddings, embed_path, name, view, label_id)
                else:
                    # gt_sitk = sitk.ReadImage(path_to_gt)
                    # gt_data = sitk.GetArrayFromImage(gt_sitk)
                    gt_data, _ = loader(path_to_gt)
                    gt_data = gt_data.array

                    # Taking only the mask for label id
                    gt_data[gt_data > label_id] = 0
                    gt_data[gt_data < label_id] = 0

                    # Binarizing mask
                    gt_data[gt_data > 0] = 1
                    assert np.max(gt_data) == 1 and np.unique(gt_data).shape[0] == 2, 'ground truth should be binary'

                    if np.sum(gt_data) > 1000:
                        image_data_pre, axial_index, cor_index, sag_index = self._getIndexes(self, image_data, gt_data)
                        for view, indexes in zip(['axial', 'sagittal', 'coronal'], [axial_index, sag_index, cor_index]):
                            # checking if image already exists
                            if os.path.exists(os.path.join(embed_path, name + f'_{view}_label_id_{label_id}.npz')):
                                logger.info(f'SAM embeddings already computed for volume {name} on label index {label_id} and {view} view')
                                continue
                            else:
                                idx_min, idx_max = np.min(indexes), np.max(indexes)
                                imgs, gts, img_embeddings = self._preprocess_img(self, image_data_pre, idx_min, idx_max, view, gt_data)
                                self._save_npz(imgs, gts, img_embeddings, embed_path, name, view, label_id)
                    else:
                        logger.warning(f'Label id {label_id} in image {name} is too small for training')

            logger.info(f"Embedding computation complete!")

    @staticmethod
    def _getIndexes(self, image_data, gt_data):
        # nii preprocess start
        image_data_pre = np.clip(image_data, self.lower_bound, self.upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre)) / (
                np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
        image_data_pre[image_data == 0] = 0
        image_data_pre = np.uint8(image_data_pre)
        if gt_data is not None:
            axial_index, cor_index, sag_index = np.where(gt_data > 0)
        else:
            axial_index = [0, image_data.shape[2]]
            cor_index = [0, image_data.shape[0]]
            sag_index = [0, image_data.shape[1]]
        return image_data_pre, axial_index, cor_index, sag_index

    @staticmethod
    def _computeEmbedding(self, img_slice_i):
        sam_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(img_slice_i)
        # resized_shapes.append(resize_img.shape[:2])
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        # model input: (1, 3, 1024, 1024)
        input_image = self.sam_model.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
        assert input_image.shape == (1, 3, self.sam_model.image_encoder.img_size,
                                     self.sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
        with torch.no_grad():
            embedding = self.sam_model.image_encoder(input_image)
            img_embedding = embedding.cpu().numpy()[0]
        return img_embedding

    @staticmethod
    def _preprocess_img(self, image_data_pre, idx_min, idx_max, view, gt_data):
        imgs = []
        gts = []
        img_embeddings = []
        logger.info(f'Computing embeddings for {view} view')
        for i in range(idx_min, idx_max):

            if view == 'axial':
                gt_slice_i = gt_data[:, :, i] if gt_data is not None else []
                img_slice_i = image_data_pre[:, :, i]
            elif view == 'coronal':
                gt_slice_i = gt_data[:, i, :] if gt_data is not None else []
                img_slice_i = image_data_pre[:, i, :]
            elif view == 'sagittal':
                gt_slice_i = gt_data[i, :, :] if gt_data is not None else []
                img_slice_i = image_data_pre[i, :, :]

            if len(gt_slice_i) > 0:
                gt_slice_i = transform.resize(gt_slice_i, (self.image_size, self.image_size), order=0, preserve_range=True,
                                          mode='constant', anti_aliasing=True)
                if np.sum(gt_slice_i) > 100:
                    # resize img_slice_i to 256x256
                    img_slice_i = transform.resize(img_slice_i, (self.image_size, self.image_size), order=3,
                                                   preserve_range=True, mode='constant', anti_aliasing=True)
                    # convert to three channels
                    img_slice_i = np.uint8(np.repeat(img_slice_i[:, :, None], 3, axis=-1))
                    assert len(img_slice_i.shape) == 3 and img_slice_i.shape[2] == 3, 'image should be 3 channels'
                    assert img_slice_i.shape[0] == gt_slice_i.shape[0] and img_slice_i.shape[1] == gt_slice_i.shape[
                        1], 'image and ground truth should have the same size'
                    imgs.append(img_slice_i)
                    assert np.sum(gt_slice_i) > 100, 'ground truth should have more than 100 pixels'
                    gts.append(gt_slice_i)
                    img_embeddings.append(self._computeEmbedding(self, img_slice_i))
            else:
                # resize img_slice_i to 256x256
                img_slice_i = transform.resize(img_slice_i, (self.image_size, self.image_size), order=3,
                                               preserve_range=True, mode='constant', anti_aliasing=True)
                # convert to three channels
                img_slice_i = np.uint8(np.repeat(img_slice_i[:, :, None], 3, axis=-1))
                assert len(img_slice_i.shape) == 3 and img_slice_i.shape[2] == 3, 'image should be 3 channels'
                imgs.append(img_slice_i)
                img_embeddings.append(self._computeEmbedding(self, img_slice_i))

        return imgs, gts, img_embeddings

    @staticmethod
    def _save_npz(imgs, gts, img_embeddings, embed_path, name, view, label_id):
        # save to npz file
        if len(imgs) > 1:
            imgs = np.stack(imgs, axis=0)  # (n, 256, 256, 3)
            img_embeddings = np.stack(img_embeddings, axis=0)  # (n, 1, 256, 64, 64)
            if len(gts) > 0:
                npz_path = os.path.join(embed_path, name + f'_{view}_label_id_{label_id}.npz')
                gts = np.stack(gts, axis=0)  # (n, 256, 256)
                np.savez_compressed(npz_path, imgs=imgs, gts=gts, img_embeddings=img_embeddings)
            else:
                npz_path = os.path.join(embed_path, name + f'_{view}.npz')
                np.savez_compressed(npz_path, imgs=imgs, img_embeddings=img_embeddings)

            # save an example image for sanity check
            idx = np.random.randint(0, imgs.shape[0])
            img_idx = imgs[idx, :, :, :]
            if len(gts) > 0:
                gt_idx = gts[idx, :, :]
                bd = segmentation.find_boundaries(gt_idx, mode='inner')
                img_idx[bd, :] = [255, 0, 0]
                img_idx = ndimage.rotate(img_idx, 90)
                io.imsave(os.path.join(embed_path, name + f'_{view}_label_id_{label_id}.png'), img_idx, check_contrast=False)
            else:
                img_idx = ndimage.rotate(img_idx, 90)
                io.imsave(os.path.join(embed_path, name + f'_{view}.png'), img_idx, check_contrast=False)
