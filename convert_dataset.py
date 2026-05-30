import os
import json
from PIL import Image

# =====================================
# PATHS
# =====================================

IMG_DIR = "dataset/train/img"

ANN_DIR = "dataset/train/ann"

OUTPUT_DIR = "dataset/cnn_dataset"

# =====================================
# CREATE OUTPUT FOLDER
# =====================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("Checking folders...")

print(
    "Images folder exists:",
    os.path.exists(IMG_DIR)
)

print(
    "Annotations folder exists:",
    os.path.exists(ANN_DIR)
)

if not os.path.exists(IMG_DIR):

    raise Exception(
        f"Missing folder: {IMG_DIR}"
    )

if not os.path.exists(ANN_DIR):

    raise Exception(
        f"Missing folder: {ANN_DIR}"
    )

ann_files = os.listdir(
    ANN_DIR
)

print(
    "Annotation files found:",
    len(ann_files)
)

total_crops = 0

# =====================================
# PROCESS FILES
# =====================================

for ann_file in ann_files:

    if not ann_file.endswith(".json"):

        continue

    json_path = os.path.join(
        ANN_DIR,
        ann_file
    )

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # IMPORTANT FIX:
    # BloodImage_00001.jpeg.json
    # becomes
    # BloodImage_00001.jpeg

    image_name = os.path.splitext(
        ann_file
    )[0]

    image_path = os.path.join(
        IMG_DIR,
        image_name
    )

    if not os.path.exists(
        image_path
    ):

        print(
            "Missing image:",
            image_name
        )

        continue

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except:

        print(
            "Cannot open:",
            image_name
        )

        continue

    counter = 0

    objects = data.get(
        "objects",
        []
    )

    for obj in objects:

        label = obj.get(
            "classTitle",
            None
        )

        if label is None:

            continue

        try:

            exterior = obj[
                "points"
            ][
                "exterior"
            ]

            x1, y1 = exterior[0]

            x2, y2 = exterior[1]

        except:

            continue

        # avoid invalid boxes

        if x2 <= x1 or y2 <= y1:

            continue

        crop = image.crop(
            (
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            )
        )

        save_dir = os.path.join(
            OUTPUT_DIR,
            label
        )

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        save_name = (
            f"{os.path.splitext(image_name)[0]}"
            f"_{counter}.jpg"
        )

        save_path = os.path.join(
            save_dir,
            save_name
        )

        crop.save(
            save_path
        )

        counter += 1

        total_crops += 1

print()

print("================================")

print(
    "Dataset conversion completed!"
)

print(
    "Total crops created:",
    total_crops
)

print(
    "Saved to:",
    OUTPUT_DIR
)

print("================================")