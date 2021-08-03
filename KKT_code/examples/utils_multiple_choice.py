# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function


import logging
import os
import sys
from io import open
import json
import csv
import glob
import tqdm
from typing import List
from transformers import PreTrainedTokenizer
import random
# from mctest import parse_mc
#from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, contexts_knowledge=None,sim_content=None,qa_knowledge=None,label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts # list of str
        self.contexts_knowledge=contexts_knowledge # list of list[each utt's knowledge_id]
        self.endings = endings
        self.sim_content=sim_content # list of list[[utt_id,utt_score],[utt_id,utt_score]...[utt_id,utt_score]]
        self.qa_knowledge=qa_knowledge # list of list[each utt's knowledge_id],length is 3
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label
                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': attention_mask,
                'token_type_ids': token_type_ids,
                't_pq_end_pos':t_pq_end_pos,
                'pivots_mask':pivots_mask,
                'qa_knowledge_ids':qa_knowledge_ids,
                'qa_knowledge_mask':qa_knowledge_mask,
                'contexts_knowledge_ids':contexts_knowledge_ids,
                'contexts_knowledge_mask':contexts_knowledge_mask
            }
            for input_ids, attention_mask,
                token_type_ids,t_pq_end_pos,
                pivots_mask,qa_knowledge_ids,
                qa_knowledge_mask,contexts_knowledge_ids,
                contexts_knowledge_mask in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class DreamProcessor(DataProcessor):
    """Processor for the SWAG data set."""
    def __init__(self):
        self.data_pos={"train":0,"dev":1,"test":2}
        self.D = [[], [], []]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""

        if len(self.D[self.data_pos[type]])==0:
            random.seed(42)
            for sid in range(3):
                with open([data_dir + "/" + "train.json", data_dir + "/"  + "dev.json",
                           data_dir + "/" + "test.json"][sid], "r") as f:
                    data = json.load(f)
                    if sid == 0:
                        random.shuffle(data)
                    for i in range(len(data)):
                        contexts=[x.lower() for x in data[i][0]]
                        contexts_knowledge=data[i][3]
                        for j in range(len(data[i][1])):
                            d=[contexts,contexts_knowledge]
                            d.append(data[i][1][j]["question"].lower())
                            for k in range(3):
                                d += [data[i][1][j]["choice"][k].lower()]
                            d += [data[i][1][j]["answer"].lower()]
                            for k in range(3):
                                d += [data[i][1][j]["utt_rank"][k]]
                            for k in range(3):
                                d += [data[i][1][j]["knowledge"][k]]
                            self.D[sid] += [d]
        data=self.D[self.data_pos[type]]
        examples = []
        for (i, d) in enumerate(data):
            for k in range(3):
                if data[i][3 + k] == data[i][6]:
                    answer = str(k)

            label = answer
            guid = "%s-%s-%s" % (type, i, k)

            examples.append(
                InputExample(example_id=guid,
                             contexts=d[0],
                             contexts_knowledge=d[1],
                             question=d[2],
                             endings=[d[3],d[4],d[5]],
                             sim_content=[d[7],d[8],d[8]],
                             qa_knowledge=[d[10],d[11],d[12]],
                             label=label))

        return examples


def get_contexts_info(contexts_knowledge,max_contexts_knowledge,max_length):
    contexts_knowledge_ids=[-1]*max_length
    contexts_knowledge_mask=[0]*max_length
    count=0
    for i in range(20):
        for j in range(len(contexts_knowledge)):
            kid=contexts_knowledge[j][i]
            if kid==-1:
                continue
            if count==max_contexts_knowledge:
                return contexts_knowledge_ids,contexts_knowledge_mask
            contexts_knowledge_ids[count]=kid
            contexts_knowledge_mask[count]=1
            count+=1
    return contexts_knowledge_ids,contexts_knowledge_mask


def get_pivots_mask(contexts_utt_length,t_qa_len,max_length,sim_content,top_k):
    all_utts_starts=[1]
    for utt_length in contexts_utt_length:
        all_utts_starts.append(all_utts_starts[-1]+utt_length) #最后一位为最后的一个可能的sep位置，其余皆为每句的开头位置
    pivots_utts=sorted([i[0] for i in sim_content[:top_k]])
    pivots_mask=[0]*max_length
    for utt_id in pivots_utts:
        utt_start=all_utts_starts[utt_id]
        next_utt_start=all_utts_starts[utt_id+1]
        if utt_start>=max_length-2-t_qa_len:
            break
        if next_utt_start>max_length-2-t_qa_len:
            pivots_mask[utt_start:max_length-2-t_qa_len]=[1]*(max_length-2-t_qa_len-utt_start)
            break
        pivots_mask[utt_start:next_utt_start]=[1]*(next_utt_start-utt_start)
    return pivots_mask


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    top_k:int,
    max_contexts_knowledge:int,
    model_type:str,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
    truncation_strategy='longest_first',
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    if model_type=="albert_all":
        for (ex_index, example) in enumerate(tqdm.tqdm(examples, desc="convert examples to features")):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            choices_features = []
            contexts=example.contexts
            contexts_knowledge=example.contexts_knowledge
            contexts_utt_length =[len(tokenizer.tokenize(x)) for x in contexts]
            contexts_knowledge_ids,contexts_knowledge_mask=get_contexts_info(contexts_knowledge,max_contexts_knowledge,max_length)
            joint_contexts = '\n'.join(contexts)
            for ending_idx, (ending,sim_content,qa_knowledge) in enumerate(zip(example.endings,example.sim_content,example.qa_knowledge)):
                qa = example.question + " " + ending

                t_qa_len=len(tokenizer.tokenize(qa))
                context_max_len=max_length-3-t_qa_len
                t_c_len=len(tokenizer.tokenize(joint_contexts))
                if t_c_len>context_max_len:
                    t_c_len=context_max_len

                assert(t_qa_len+t_c_len+3<=max_length)

                inputs = tokenizer.encode_plus(
                    joint_contexts,
                    qa,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation_strategy=truncation_strategy
                )


                input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

                assert(len(input_ids[t_c_len+t_qa_len:])==3)

                t_pq_end_pos=[t_c_len, t_c_len + 1 + t_qa_len] # [CLS] CONTEXT [SEP] QUESTION OPTION [SEP]

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                pad_token=tokenizer.pad_token_id

                # Zero-pad up to the sequence length.
                padding_length = max_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                    token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                qa_knowledge_ids=[-1]*max_length
                count=0
                for (i,kid) in enumerate(qa_knowledge):
                    if kid == -1:
                        break
                    count+=1
                    qa_knowledge_ids[i]=kid
                qa_knowledge_mask=[1]*count+[0]*(max_length-count)
                pivots_mask=get_pivots_mask(contexts_utt_length,t_qa_len,max_length,sim_content,top_k)

                assert len(input_ids) == max_length
                assert len(attention_mask) == max_length
                assert len(token_type_ids) == max_length
                assert len(qa_knowledge_ids) == max_length
                assert len(qa_knowledge_mask) == max_length
                assert len(pivots_mask) == max_length
                assert len(contexts_knowledge_ids) == max_length
                assert len(contexts_knowledge_mask) == max_length

                choices_features.append((input_ids, attention_mask,
                                         token_type_ids,t_pq_end_pos,
                                         pivots_mask,qa_knowledge_ids,
                                         qa_knowledge_mask,contexts_knowledge_ids,
                                         contexts_knowledge_mask))

            label = label_map[example.label]

            '''
            if ex_index < 2:
                logger.info("*** Example ***")
                logger.info("dream_id: {}".format(example.example_id))
                logger.info("label: {}".format(label))
                for choice_idx, (input_ids, attention_mask, token_type_ids,sep_pos,pivots_mask,qa_knowledge_ids,qa_knowledge_mask,
                                 contexts_knowledge_ids,contexts_knowledge_mask) in enumerate(choices_features):
                    logger.info("choice: {}".format(choice_idx))
                    logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                    logger.info("attention_mask: {}".format(' '.join(map(str, attention_mask))))
                    logger.info("token_type_ids: {}".format(' '.join(map(str, token_type_ids))))
                    logger.info("pivots_mask: {}".format(' '.join(map(str, pivots_mask))))
                    logger.info("qa_knowledge_ids: {}".format(' '.join(map(str, qa_knowledge_ids))))
                    logger.info("qa_knowledge_mask: {}".format(' '.join(map(str, qa_knowledge_mask))))
                    logger.info("contexts_knowledge_ids: {}".format(' '.join(map(str, contexts_knowledge_ids))))
                    logger.info("contexts_knowledge_mask: {}".format(' '.join(map(str, contexts_knowledge_mask))))
            '''
            features.append(
                InputFeatures(
                    example_id=example.example_id,
                    choices_features=choices_features,
                    label=label,
                )
            )
    elif model_type=="albert_pivots":
        for (ex_index, example) in enumerate(tqdm.tqdm(examples, desc="convert examples to features")):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            choices_features = []
            contexts = example.contexts
            contexts_knowledge = example.contexts_knowledge

            for ending_idx, (ending, sim_content, qa_knowledge) in enumerate(
                    zip(example.endings, example.sim_content, example.qa_knowledge)):
                pivots_utt_ids=sorted([x[0] for x in sim_content[:top_k]])
                pivots=[contexts[i] for i in pivots_utt_ids]
                pivots_knowledge=[contexts_knowledge[i] for i in pivots_utt_ids]
                pivots_knowledge_ids, pivots_knowledge_mask = get_contexts_info(pivots_knowledge,
                                                                                    max_contexts_knowledge, max_length)
                joint_pivots="\n".join(pivots)
                qa = example.question + " " + ending

                t_qa_len = len(tokenizer.tokenize(qa))
                context_max_len = max_length - 3 - t_qa_len
                t_c_len = len(tokenizer.tokenize(joint_pivots))
                if t_c_len > context_max_len:
                    t_c_len = context_max_len

                assert (t_qa_len + t_c_len + 3 <= max_length)

                inputs = tokenizer.encode_plus(
                    joint_pivots,
                    qa,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation_strategy=truncation_strategy
                )

                input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

                assert (len(input_ids[t_c_len + t_qa_len:]) == 3)

                t_pq_end_pos = [t_c_len, t_c_len + 1 + t_qa_len]  # [CLS] CONTEXT [SEP] QUESTION OPTION [SEP]

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                pad_token = tokenizer.pad_token_id

                # Zero-pad up to the sequence length.
                padding_length = max_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                    token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                qa_knowledge_ids = [-1] * max_length
                count = 0
                for (i, kid) in enumerate(qa_knowledge):
                    if kid == -1:
                        break
                    count += 1
                    qa_knowledge_ids[i] = kid
                qa_knowledge_mask = [1] * count + [0] * (max_length - count)
                pivots_mask = [0]

                assert len(input_ids) == max_length
                assert len(attention_mask) == max_length
                assert len(token_type_ids) == max_length
                assert len(qa_knowledge_ids) == max_length
                assert len(qa_knowledge_mask) == max_length
                assert len(pivots_mask) == 1
                assert len(pivots_knowledge_ids) == max_length
                assert len(pivots_knowledge_mask) == max_length

                choices_features.append((input_ids, attention_mask,
                                         token_type_ids, t_pq_end_pos,
                                         pivots_mask, qa_knowledge_ids,
                                         qa_knowledge_mask, pivots_knowledge_ids,
                                         pivots_knowledge_mask))

            label = label_map[example.label]

            '''
            if ex_index < 2:
                logger.info("*** Example ***")
                logger.info("dream_id: {}".format(example.example_id))
                logger.info("label: {}".format(label))
                for choice_idx, (input_ids, attention_mask, token_type_ids,sep_pos,pivots_mask,qa_knowledge_ids,qa_knowledge_mask,
                                 contexts_knowledge_ids,contexts_knowledge_mask) in enumerate(choices_features):
                    logger.info("choice: {}".format(choice_idx))
                    logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                    logger.info("attention_mask: {}".format(' '.join(map(str, attention_mask))))
                    logger.info("token_type_ids: {}".format(' '.join(map(str, token_type_ids))))
                    logger.info("pivots_mask: {}".format(' '.join(map(str, pivots_mask))))
                    logger.info("qa_knowledge_ids: {}".format(' '.join(map(str, qa_knowledge_ids))))
                    logger.info("qa_knowledge_mask: {}".format(' '.join(map(str, qa_knowledge_mask))))
                    logger.info("contexts_knowledge_ids: {}".format(' '.join(map(str, contexts_knowledge_ids))))
                    logger.info("contexts_knowledge_mask: {}".format(' '.join(map(str, contexts_knowledge_mask))))
            '''
            features.append(
                InputFeatures(
                    example_id=example.example_id,
                    choices_features=choices_features,
                    label=label,
                )
            )

    return features



processors = {
    "dream": DreamProcessor,
}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {
    "race", 4,
    "swag", 4,
    "arc", 4,
    "dream", 3,
    "mctest", 4
}
