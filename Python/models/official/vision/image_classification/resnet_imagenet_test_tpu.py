# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Test the keras ResNet model with ImageNet data on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.utils.misc import keras_utils
from official.utils.testing import integration
from official.vision.image_classification import imagenet_preprocessing
from official.vision.image_classification import resnet_imagenet_main


class KerasImagenetTest(tf.test.TestCase):
  """Unit tests for Keras ResNet with ImageNet."""

  _extra_flags = [
      "-batch_size", "4",
      "-train_steps", "1",
      "-use_synthetic_data", "true"
  ]
  _tempdir = None

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(KerasImagenetTest, cls).setUpClass()
    resnet_imagenet_main.define_imagenet_keras_flags()

  def setUp(self):
    super(KerasImagenetTest, self).setUp()
    imagenet_preprocessing.NUM_IMAGES["validation"] = 4

  def tearDown(self):
    super(KerasImagenetTest, self).tearDown()
    tf.io.gfile.rmtree(self.get_temp_dir())

  def test_end_to_end_tpu(self):
    """Test Keras model with TPU distribution strategy."""
    config = keras_utils.get_config_proto_v1()
    tf.compat.v1.enable_eager_execution(config=config)

    extra_flags = [
        "-distribution_strategy", "tpu",
        "-data_format", "channels_last",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_tpu_bf16(self):
    """Test Keras model with TPU and bfloat16 activation."""
    config = keras_utils.get_config_proto_v1()
    tf.compat.v1.enable_eager_execution(config=config)

    extra_flags = [
        "-distribution_strategy", "tpu",
        "-data_format", "channels_last",
        "-dtype", "bf16",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
