# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""We calculate n-Grams from the training text. We will use this as an
evaluation metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange


def hash_function(input_tuple):
  """Hash function for a tuple."""
  return hash(input_tuple)


def find_all_ngrams(dataset, n):
  """Generate a list of all ngrams."""
  return zip(*[dataset[i:] for i in xrange(n)])


def construct_ngrams_dict(ngrams_list):
  """Construct a ngram dictionary which maps an ngram tuple to the number
  of times it appears in the text."""
  counts = {}

  for t in ngrams_list:
    key = hash_function(t)
    if key in counts:
      counts[key] += 1
    else:
      counts[key] = 1
  return counts


def percent_unique_ngrams_in_train(train_ngrams_dict, gen_ngrams_dict):
  """Compute the percent of ngrams generated by the model that are
  present in the training text and are unique."""

  # *Total* number of n-grams produced by the generator.
  total_ngrams_produced = 0

  for _, value in gen_ngrams_dict.iteritems():
    total_ngrams_produced += value

  # The unique ngrams in the training set.
  unique_ngrams_in_train = 0.

  for key, _ in gen_ngrams_dict.iteritems():
    if key in train_ngrams_dict:
      unique_ngrams_in_train += 1
  return float(unique_ngrams_in_train) / float(total_ngrams_produced)
