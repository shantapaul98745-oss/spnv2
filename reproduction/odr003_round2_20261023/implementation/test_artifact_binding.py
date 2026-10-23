"""Metadata-only regression for actual frozen prepared-artifact records."""
import copy
import unittest
from unittest.mock import patch

from artifact_binding import same_artifact, require_same_artifact


class ArtifactBindingTests(unittest.TestCase):
    def setUp(self):
        self.binding = dict(
            path="/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/"
                 "odr003_exploratory_20261023/data/lightbox/stream_2021.json",
            sha256="37bd346bc597c354fe2e17198aa595363faf3f67a7139ad2f4f91e73d90b4bbe")
        self.prepared = dict(self.binding, bytes=215271)

    def test_actual_prepared_bytes_field_is_informational(self):
        require_same_artifact(self.binding, self.prepared, "boundary differs")
        self.assertTrue(same_artifact(self.binding, self.prepared))
        require_same_artifact(self.binding, self.binding, "boundary differs")
        for count in (0, 215271, 999999):
            require_same_artifact(self.binding, dict(self.binding, bytes=count), "boundary differs")

    def test_different_literal_path_or_digest_is_rejected(self):
        for key, value in (("path", self.binding["path"].replace("2021", "2022")),
                           ("sha256", "0" * 64)):
            artifact = dict(self.prepared, **{key: value})
            self.assertFalse(same_artifact(self.binding, artifact))
            with self.assertRaisesRegex(ValueError, "boundary differs"):
                require_same_artifact(self.binding, artifact, "boundary differs")

    def test_literal_case_change_is_not_normalized_away(self):
        artifact = dict(self.prepared, path=self.binding["path"].replace("DATA", "data"))
        self.assertFalse(same_artifact(self.binding, artifact))

    def test_schema_cannot_add_task_metadata_or_unknown_proof_fields(self):
        cases = ((self.prepared, self.prepared),
                 (self.binding, dict(self.prepared, ignored="anything")),
                 ({"path": self.binding["path"]}, self.prepared),
                 (self.binding, {"sha256": self.binding["sha256"]}),
                 (None, self.prepared), (self.binding, []))
        for binding, artifact in cases:
            with self.subTest(binding=binding, artifact=artifact), self.assertRaises(ValueError):
                same_artifact(binding, artifact)

    def test_bad_byte_count_cannot_be_hidden_as_informational(self):
        for count in (-1, True, 1.0, "215271", None):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "byte count"):
                same_artifact(self.binding, dict(self.prepared, bytes=count))

    def test_invalid_digest_rejected_on_both_sides(self):
        for digest in (None, 1, "a" * 63, "a" * 65, "G" * 64, "A" * 64, "a" * 64 + "\n"):
            for side in ("binding", "artifact"):
                binding, artifact = copy.deepcopy(self.binding), copy.deepcopy(self.prepared)
                (binding if side == "binding" else artifact)["sha256"] = digest
                with self.subTest(digest=digest, side=side), self.assertRaisesRegex(ValueError, "sha256"):
                    same_artifact(binding, artifact)

    def test_invalid_path_is_not_resolved_into_an_allowed_identity(self):
        for path in (None, 1, "", "relative.json", "/", "//home/file.json", "/home//file.json",
                     "/home/./file.json", "/home/../file.json", "/home/file.json/",
                     "/home/file\x00.json", "/home/file\n.json", "/home\\file.json",
                     "C:/home/file.json"):
            for side in ("binding", "artifact"):
                binding, artifact = copy.deepcopy(self.binding), copy.deepcopy(self.prepared)
                (binding if side == "binding" else artifact)["path"] = path
                with self.subTest(path=path, side=side), self.assertRaisesRegex(ValueError, "path"):
                    same_artifact(binding, artifact)

    def test_label_identity_never_opens_or_resolves_the_label(self):
        label = dict(path="/sealed/data/lightbox/dev_labels.json", sha256="7b61ca380e3f004a0e78f8035fec65a60a3b5db420f697dbf23512cf917c7f3d")
        artifact = dict(label, bytes=797482)
        with patch("builtins.open", side_effect=AssertionError("sealed GT opened")), \
             patch("io.open", side_effect=AssertionError("sealed GT opened")), \
             patch("os.stat", side_effect=AssertionError("sealed GT statted")), \
             patch("pathlib.Path.resolve", side_effect=AssertionError("sealed GT resolved")):
            require_same_artifact(label, artifact, "GT boundary changed")
            with self.assertRaisesRegex(ValueError, "GT boundary changed"):
                require_same_artifact(label, dict(artifact, sha256="0" * 64), "GT boundary changed")


if __name__ == "__main__":
    unittest.main()
