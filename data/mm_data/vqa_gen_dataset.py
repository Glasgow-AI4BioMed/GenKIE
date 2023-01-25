# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    src_bboxes = merge("source_bboxes")
    B, N = src_tokens.shape
    resized_bboxes = torch.zeros(B, N, 8, device=src_tokens.device, dtype=torch.long)
    for i in range(B):
        tmp_bbox = src_bboxes[i]
        max_x, max_y = torch.max(tmp_bbox[:, ::2]), torch.max(tmp_bbox[:, 1::2])
        tmp_bbox[:, ::2] = (tmp_bbox[:, ::2] / max_x) * 1023
        tmp_bbox[:, 1::2] = (tmp_bbox[:, 1::2] / max_y) * 1023
        resized_bboxes[i] = tmp_bbox.to(torch.long)
    src_bboxes = torch.clip(resized_bboxes, 0, 1023)

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    conf = None
    if samples[0].get("conf", None) is not None:
        conf = torch.cat([s['conf'] for s in samples], dim=0)

    ref_dict = None
    if samples[0].get("ref_dict", None) is not None:
        ref_dict = np.array([s['ref_dict'] for s in samples])

    constraint_masks = None
    if samples[0].get("constraint_mask", None) is not None:
        constraint_masks = merge("constraint_mask")

    decoder_prompts = None
    if samples[0].get("decoder_prompt", None) is not None:
        decoder_prompts = np.array([s['decoder_prompt'].tolist() for s in samples])

    prefix_tokens = None
    if samples[0].get("decoder_prompt", None) is not None:
        prefix_tokens = merge("decoder_prompt")
        prefix_tokens = prefix_tokens[:, 1:]

        # prefix_bboxes = merge("prefix_bboxes")
        # B, N = prefix_tokens.shape
        # resized_bboxes = torch.zeros(B, N, 8, device=prefix_tokens.device, dtype=torch.long)
        # for i in range(B):
        #     tmp_bbox = prefix_bboxes[i]
        #     max_x, max_y = torch.max(tmp_bbox[:, ::2]), torch.max(tmp_bbox[:, 1::2])
        #     tmp_bbox[:, ::2] = (tmp_bbox[:, ::2] / max_x) * 1023
        #     tmp_bbox[:, 1::2] = (tmp_bbox[:, 1::2] / max_y) * 1023
        #     resized_bboxes[i] = tmp_bbox.to(torch.long)
        # prefix_bboxes = torch.clip(resized_bboxes, 0, 1023)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        )
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "src_bboxes": src_bboxes,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "conf": conf,
        "ref_dict": ref_dict,
        "constraint_masks": constraint_masks,
        "decoder_prompts": decoder_prompts,
        "target": target,
        "prefix_tokens": prefix_tokens
        # "prefix_bboxes": prefix_bboxes
    }

    return batch


class VqaGenDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_object_length=30,
        max_tgt_length=30,
        patch_image_size=224,
        add_object=False,
        constraint_trie=None,
        imagenet_default_mean_and_std=False,
        prompt_type="none"
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_object_length = max_object_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size

        self.add_object = add_object
        self.constraint_trie = constraint_trie
        self.prompt_type = prompt_type

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        item = self.dataset[index]
        # if len(item) == 5:
        predict_objects = None
        if len(item) == 4:
            # uniq_id, image, question, ref, predict_objects = item
            uniq_id, image, question_pos, ref = item
        else:
            uniq_id, image, question, ref, predict_objects, caption = item

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        q_p = question_pos.split('\dsep')
        q, p = [], []
        for qp in q_p:
            q.append(qp.split('\sep')[0])
            p.append(qp.split('\sep')[1])
        assert len(p) == len(q)
        src_item, src_bboxes = [], []
        for i, question in enumerate(q):
            # question = self.pre_question(question, self.max_src_length)
            question = self.pre_question(question)
        # question = self.pre_question(question_pos)
            if i == len(q) - 1:
                question = question + '?' if not question.endswith('?') else question
        # src_item = self.encode_text(' {}'.format(question))
            encoded_res = self.encode_text(' {}'.format(question))
            src_item.extend(encoded_res)
            # if i == len(q) - 1 or i == len(q) - 2:
            if p[i] == '':
                assert p[i] == ''
                bbox = [0, 0, 0, 0, 0, 0, 0, 0]
                src_bboxes.extend([bbox.copy() for times in range(len(encoded_res))])
            else:
                try:
                    bbox = list(map(int, p[i].split(',')))
                    src_bboxes.extend([bbox.copy() for times in range(len(encoded_res))])
                except ValueError as e:
                    print('value error')
        src_item = torch.as_tensor(src_item)
        src_bboxes.insert(0, [0, 0, 0, 0, 0, 0, 0, 0])
        src_bboxes.append([0, 0, 0, 0, 0, 0, 0, 0])
        src_bboxes = torch.as_tensor(src_bboxes)
        # ref_dict = {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in ref.split('&&')}
        # answer = max(ref_dict, key=ref_dict.get)
        # conf = torch.tensor([ref_dict[answer]])
        # tgt_item = self.encode_text(" {}".format(answer))
        tgt_item = self.encode_text(" {}".format(ref))

        if self.add_object and predict_objects is not None:
            predict_object_seq = ' '.join(predict_objects.strip().split('&&')[:self.max_object_length])
            predict_object_item = self.encode_text(" object: {}".format(predict_object_seq))
            src_item = torch.cat([src_item, predict_object_item])
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        if self.prompt_type == 'none':
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = self.bos_item
            # decoder_prompt = torch.cat([self.bos_item, self.encode_text(" {}".format(prefix_str))])
            # decoder_prompt, prefix_bboxes = [], []
            #
            # prefix_q_p = prefix_str.split('\dsep')
            # prefix_q, prefix_p = [], []
            # for qp in prefix_q_p:
            #     prefix_q.append(qp.split('\sep')[0])
            #     prefix_p.append(qp.split('\sep')[1])
            # assert len(p) == len(q)
            # src_item, src_bboxes = [], []
            # for i, question in enumerate(prefix_q):
            #     # question = self.pre_question(question, self.max_src_length)
            #     question = self.pre_question(question)
            #     # if i == len(q) - 1:
            #     #     question = question + '?' if not question.endswith('?') else question
            #     # src_item = self.encode_text(' {}'.format(question))
            #     encoded_res = self.encode_text(' {}'.format(question))
            #     decoder_prompt.extend(encoded_res)
            #     if i == len(prefix_q) - 1:
            #         assert prefix_p[i] == ''
            #         bbox = [0, 0, 0, 0, 0, 0, 0, 0]
            #         prefix_bboxes.extend([bbox.copy() for times in range(len(encoded_res))])
            #     else:
            #         bbox = list(map(int, prefix_p[i].split(',')))
            #         prefix_bboxes.extend([bbox.copy() for times in range(len(encoded_res))])
            # decoder_prompt = torch.as_tensor(decoder_prompt)
            # decoder_prompt = torch.cat([self.bos_item, decoder_prompt])
            # prefix_bboxes.insert(0, [0, 0, 0, 0, 0, 0, 0, 0])
            # prefix_bboxes = torch.as_tensor(prefix_bboxes)
        elif self.prompt_type == 'src':
            prev_output_item = torch.cat([src_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item
        elif self.prompt_type == 'prev_output':
            prev_output_item = torch.cat([src_item[:-1], tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item[:-1]
        else:
            raise NotImplementedError
        target_item[:-len(tgt_item)-1] = self.tgt_dict.pad()

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "decoder_prompt": decoder_prompt,
            "source_bboxes": src_bboxes
            # "prefix_bboxes": prefix_bboxes
            # "ref_dict": ref_dict,
            # "conf": conf,
        }
        if self.constraint_trie is not None:
            constraint_mask = torch.zeros((len(target_item), len(self.tgt_dict))).bool()
            start_idx = len(target_item) - len(tgt_item) - 1
            for i in range(len(target_item)-len(tgt_item)-1, len(target_item)):
                constraint_prefix_token = [self.tgt_dict.bos()] + target_item[start_idx:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            example["constraint_mask"] = constraint_mask
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
