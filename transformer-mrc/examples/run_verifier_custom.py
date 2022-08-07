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
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tqdm import tqdm

import argparse
import collections
import json
import os
import pandas as pd
import random

def get_score1(args):
    cof = [1, 1]
    best_cof = [1]

    with open(args.predict_file,'r') as f:
        all_data = json.load(f)

    input_data = []
    if args.input_null_files:
        for input_file in args.input_null_files.split(","):
            with open(input_file, 'r') as reader:
                input_data.append(json.load(reader, strict=False))

    if args.input_nbest_files:
        idx = 0
        all_nbest = collections.OrderedDict()
        for input_file in args.input_nbest_files.split(","):
            with open(input_file, "r") as reader:
                nbest_data = json.load(reader, strict=False)
                for (key, entries) in nbest_data.items():
                    if key not in all_nbest:
                        all_nbest[key] = collections.defaultdict(float)
                    for entry in entries:
                        all_nbest[key][entry["text"]] += best_cof[idx] * entry["probability"]
            idx += 1
        output_predictions = {}
        for (key, entry_map) in all_nbest.items():
            sorted_texts = sorted(
                entry_map.keys(), key=lambda x: entry_map[x], reverse=True)
            best_text = sorted_texts[0]
            output_predictions[key] = best_text

    questions_flat = {}
    training_data = []
    score_fields = [os.path.basename(input_file).split('.')[0] for input_file in args.input_null_files.split(",")]
    for doc in tqdm(all_data['data']):
        for paragraph in doc['paragraphs']:
            for qas in paragraph['qas']:
                if qas['answers']:
                    key = qas['id']
                    qas.update({
                        'context': paragraph['context'],
                        'context_len': paragraph['context_len'],
                        'answer_text': qas['answers'][0]['text'],
                        'answer_len': qas['answers'][0]['answer_len'],
                        'answer_start': qas['answers'][0]['answer_start']
                    })

                    valid = True
                    valid = valid and paragraph['context_len'] >= args.min_context_len if args.min_context_len else valid
                    valid = valid and qas['answer_len'] >= args.min_answer_len if args.min_answer_len else valid

                    qas.update({'best_prediction': output_predictions[key]}) if args.input_nbest_files else None
                    if args.input_null_files:
                        score = 0
                        for idx, score_field in enumerate(score_fields):
                            qas.update({score_field:input_data[idx][key]})
                            score += input_data[idx][key]
                            valid = valid and input_data[idx][key] < 0
                        score /= len(score_fields)
                        qas.update({'average_score':score})
                        valid = valid and qas['average_score'] <= args.threshold if args.threshold and args.input_null_files else valid

                    if valid:
                        training_data.append({
                            'question': qas['question'],
                            'id': qas['id'],
                            'answers': [ans['text'] for ans in qas['answers']],
                            'neg_answers': [ans['text'] for ans in qas['neg_answers']] if 'neg_answers' in qas else []
                        })

                    del qas['answers']
                    if 'neg_answers' in qas:
                        del qas['neg_answers']
                    questions_flat[key] = qas

    df = pd.DataFrame(questions_flat.values())
    df.to_csv(os.path.join(args.output_dir,'flat_predictions.csv'))

    random.shuffle(training_data)
    print(f"Original Dataset Total Questions: {len(questions_flat)}")
    print(f"filtered Dataset Total Questions: {len(training_data)}")
    train_size = int(len(training_data) * 0.6)
    test_size = int(len(training_data) * 0.2)

    dataset_dir = os.path.join(args.output_dir, 'finetuning_data')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    with open(os.path.join(dataset_dir, 'train_preprocessed.json'), 'w+') as f:
        json.dump({'data': training_data[:train_size]}, f, indent=1)
    with open(os.path.join(dataset_dir, 'dev_preprocessed.json'), 'w+') as f:
        json.dump({'data': training_data[train_size:][:test_size]}, f, indent=1)
    with open(os.path.join(dataset_dir, 'test_preprocessed.json'), 'w+') as f:
        json.dump({'data': training_data[train_size:][test_size:]}, f, indent=1)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--input_null_files', type=str, default="cls_score.json,null_odds.json")
    parser.add_argument('--input_nbest_files', type=str, default="nbest_predictions.json")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--predict_file", default="data/dev.json")
    parser.add_argument("--min_context_len", type=int)
    parser.add_argument("--min_answer_len", type=int)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    get_score1(args)

if __name__ == "__main__":
    main()
