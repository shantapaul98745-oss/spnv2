"""CPU contracts: GT isolation, immutable image identity and whole-group split."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import data_boundary as db


class DataBoundaryTests(unittest.TestCase):
    def test_reserve_numeric_columns_never_parsed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"test.csv"
            path.write_text("img000001.jpg," + ",".join(["0"]*35) + "\nimg000002.jpg,THIS_IS_NOT_A_NUMBER_AND_MUST_NOT_BE_PARSED\n")
            rows = db.read_developer_labels(path, "lightbox", ["lightbox/img000001.jpg"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(db.read_ids_only(path), ["img000001.jpg", "img000002.jpg"])

    def test_group_boundary_spans_domains(self):
        groups = [["lightbox/img000001.jpg", "sunlamp/img000001.jpg"],
                  ["lightbox/img000002.jpg"], ["sunlamp/img000002.jpg"]]
        selected, _ = db.choose_developer(groups, per_domain=1)
        self.assertEqual(sum(x.startswith('lightbox/') for x in selected), 1)
        self.assertEqual(sum(x.startswith('sunlamp/') for x in selected), 1)
        for group in groups:
            self.assertIn(len(set(group)&selected), [0, len(group)])

    def test_impossible_boundary_fails_instead_of_splitting(self):
        with self.assertRaisesRegex(ValueError, "scientific review"):
            db.choose_developer([["lightbox/img000001.jpg", "lightbox/img000002.jpg"], ["sunlamp/img000001.jpg"]], per_domain=1)

    def image_fixture(self, temp):
        root = Path(temp)
        path = root/"lightbox"/db.IMAGE_FOLDER/"img000001.jpg"
        path.parent.mkdir(parents=True)
        image = np.zeros((512, 768, 3), np.uint8)
        image[:,:,0] = 17; image[:,:,1] = 93; image[:,:,2] = 201
        self.assertTrue(cv2.imwrite(str(path), image))
        item = {"sample_id": "lightbox/img000001.jpg", "domain": "lightbox", "filename": path.name, "image_sha256": db.digest(path)}
        manifest = root/"images.json"
        db.save_json(manifest, db.make_manifest(root, 'test', [item]))
        return path, manifest

    def test_loader_needs_no_gt_and_preserves_original_bgr(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path, manifest = self.image_fixture(temp)
            sample = db.ImageManifestDataset(manifest)[0]
            self.assertEqual(set(sample), {'image','sample_id','domain','filename'})
            np.testing.assert_array_equal(sample['image'], cv2.imread(str(image_path), cv2.IMREAD_COLOR))

    def test_mutated_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, manifest = self.image_fixture(temp)
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'Image content changed'):
                db.ImageManifestDataset(manifest)[0]

    def test_annotations_and_path_traversal_forbidden(self):
        with tempfile.TemporaryDirectory() as temp:
            _, path = self.image_fixture(temp)
            manifest = json.loads(path.read_text())
            manifest['items'][0]['translation'] = [0,0,1]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'annotations are forbidden'):
                db.load_image_manifest(path)
        with self.assertRaises(ValueError): db.safe_filename('../test.csv')

    def test_hash_tree_matches_bruteforce(self):
        tree = db.HashTree()
        values = [0, 1, 3, 7, 15, 65535, 2**63]
        for i,v in enumerate(values): tree.add(v,i)
        for target in values:
            self.assertEqual(sorted(tree.query(target, 2)), [i for i,v in enumerate(values) if (v^target).bit_count()<=2])

    def test_whole_group_selection_reproducible(self):
        groups = [[f'{domain}/img{i:06d}.jpg'] for domain in db.DOMAINS for i in range(1,601)]
        self.assertEqual(db.choose_developer(groups), db.choose_developer(groups))
        selected, _ = db.choose_developer(groups)
        self.assertEqual(len(selected), 1024)

    def test_original_transform_equal_without_gt(self):
        import albumentations as A
        from albumentations.pytorch.transforms import ToTensorV2
        with tempfile.TemporaryDirectory() as temp:
            image_path, manifest = self.image_fixture(temp)
            transform = A.Compose([A.Normalize(mean=(.485,.456,.406), std=(.229,.224,.225)), ToTensorV2()])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            direct = transform(image=image)['image']
            through = db.ImageManifestDataset(manifest, transforms=transform)[0]['image']
            self.assertTrue(np.array_equal(direct.numpy(), through.numpy()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
