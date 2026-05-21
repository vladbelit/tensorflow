# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Python object API proto visitor."""

import enum
import inspect

from tensorflow.python.platform import googletest
from tensorflow.tools.api.lib import python_object_to_proto_visitor as visitor_lib


def _TensorFlowOwnedBase(module):
  return type('TensorFlowOwnedBase', (BaseException,), {
      '__module__': module,
      'tf_owned_member': property(lambda self: None),
  })


class PythonObjectToProtoVisitorTest(googletest.TestCase):

  def test_normalize_type_canonicalizes_runtime_type_names(self):
    self.assertEqual(
        "<class 'enum.EnumType'>",
        visitor_lib._NormalizeType("<class 'enum.EnumMeta'>"))

  def test_normalize_type_keeps_only_current_canonicalizations(self):
    self.assertEqual(
        {
            "<class 'typing._UnionGenericAlias'>": 'typing.Union',
            "<class 'enum.EnumMeta'>": "<class 'enum.EnumType'>",
        },
        visitor_lib._NORMALIZE_TYPE)

  def test_normalize_is_instance_preserves_legacy_class_names(self):
    normalizations = {
        "<class 'tensorflow.lite.python.op_hint.OpHint."
        "OpHintArgumentTracker'>": (
            "<class "
            "'tensorflow.lite.python.op_hint.OpHintArgumentTracker'>"),
        "<class 'tensorflow.python.training.monitored_session."
        "_MonitoredSession.StepContext'>": (
            "<class "
            "'tensorflow.python.training.monitored_session.StepContext'>"),
        "<class 'tensorflow.python.ops.variables.Variable.SaveSliceInfo'>": (
            "<class 'tensorflow.python.ops.variables.SaveSliceInfo'>"),
    }

    for original, normalized in normalizations.items():
      self.assertEqual(normalized, visitor_lib._NormalizeIsInstance(original))

  def test_tensorflow_owned_class_matches_mid_string_module(self):
    cls = _TensorFlowOwnedBase('third_party.py.tensorflow.python.framework')

    self.assertTrue(visitor_lib._IsTensorFlowOwnedClass(cls))

  def test_tensorflow_owned_class_matches_tensorflow_family_segments(self):
    modules = (
        'third_party.py.keras.src.layers',
        'third_party.py.tensorflow_probability.python',
        'third_party.py.tf_keras.src.engine',
    )

    for module in modules:
      cls = _TensorFlowOwnedBase(module)

      self.assertTrue(visitor_lib._IsTensorFlowOwnedClass(cls), module)

  def test_tensorflow_owned_class_rejects_embedded_token(self):
    cls = _TensorFlowOwnedBase('third_party.py.not_tensorflow.python')

    self.assertFalse(visitor_lib._IsTensorFlowOwnedClass(cls))

  def test_unstable_external_runtime_member_is_pruned(self):

    class Exported(TypeError):
      pass

    self.assertTrue(
        visitor_lib._IsUnstableExternalInheritedMember(Exported, 'args'))

  def test_direct_runtime_member_override_is_kept(self):

    class Exported(TypeError):

      def add_note(self, note):
        del note

    self.assertFalse(
        visitor_lib._IsUnstableExternalInheritedMember(Exported, 'add_note'))

  def test_tensorflow_family_inherited_member_is_kept(self):
    base = _TensorFlowOwnedBase('third_party.py.tensorflow.python.framework')
    exported = type('Exported', (base,), {'__module__': 'public_api'})

    self.assertFalse(
        visitor_lib._IsUnstableExternalInheritedMember(
            exported, 'tf_owned_member'))

  def test_enum_runtime_member_is_pruned(self):

    class Exported(enum.Enum):
      VALUE = 1

    self.assertTrue(
        visitor_lib._IsUnstableExternalInheritedMember(Exported, 'name'))

  def test_inspect_signature_runtime_member_is_pruned(self):

    class Exported(inspect.Signature):
      pass

    self.assertTrue(
        visitor_lib._IsUnstableExternalInheritedMember(Exported, 'bind'))

  def test_visitor_filters_only_unstable_external_inherited_members(self):
    base = _TensorFlowOwnedBase('third_party.py.tensorflow.python.framework')
    exported = type('Exported', (base,), {'__module__': 'public_api'})
    children = [
        ('args', BaseException.args),
        ('tf_owned_member', base.tf_owned_member),
    ]

    visitor = visitor_lib.PythonObjectToProtoVisitor()
    visitor('errors.Exported', exported, children)

    self.assertEqual(['tf_owned_member'], [name for name, _ in children])


if __name__ == '__main__':
  googletest.main()
