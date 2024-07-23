import os
import json
from datasets import DatasetBuilder, GeneratorBasedBuilder, SplitGenerator, DatasetInfo, Features, Value, Sequence, Array2D

class CustomCocoDataset(GeneratorBasedBuilder):
    def _info(self):
        return DatasetInfo(
            features=Features({
                "image_id": Value("int64"),
                "image": Value("string"),
                "width": Value("int64"),
                "height": Value("int64"),
                "objects": Sequence({
                    "category_id": Value("int64"),
                    "image_id": Value("string"),
                    "id": Value("int64"),
                    "area": Value("float64"),
                    "bbox": Array2D(dtype="float32", shape=(1, 4)),
                    "segmentation": Sequence(Sequence(Value("float32"))),
                    "iscrowd": Value("bool"),
                }),
            })
        )

    def _split_generators(self, dl_manager):
        return [
            SplitGenerator(
                name="train",
                gen_kwargs={
                    "annotation_file": "./coco_annotations.json",
                    "img_dir": "./Single-Images-With-Label",
                },
            ),
            SplitGenerator(
                name="test",
                gen_kwargs={
                    "annotation_file": "./coco_annotations.json",
                    "img_dir": "./Single-Images-With-Label",
                },
            ),
        ]

    def _generate_examples(self, annotation_file, img_dir):
        with open(annotation_file, "r") as f:
            annotations = json.load(f)

        for img_info in annotations["images"]:
            img_id = img_info["id"]
            img_path = os.path.join(img_dir, img_info["file_name"])

            objects = [
                {
                    "category_id": ann["category_id"],
                    "image_id": str(img_id),
                    "id": ann["id"],
                    "area": ann["area"],
                    "bbox": [ann["bbox"]],
                    "segmentation": ann.get("segmentation", []),
                    "iscrowd": ann.get("iscrowd", False),
                }
                for ann in annotations["annotations"]
                if ann["image_id"] == img_id
            ]

            yield img_id, {
                "image_id": img_id,
                "image": img_path,
                "width": img_info["width"],
                "height": img_info["height"],
                "objects": objects,
            }

