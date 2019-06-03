from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

coref_op_library = tf.load_op_library("externals/modified_e2e_coref/coref_kernels.so")

extract_spans = coref_op_library.extract_spans
tf.NotDifferentiable("ExtractSpans")
