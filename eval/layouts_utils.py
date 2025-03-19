from pathlib import Path
import numpy as np

gen_layouts_path = Path("eval/evaluation_set/generation")
gen_renders_path = gen_layouts_path / "renders"
gen_jsons_path = gen_layouts_path / "json"

cons_layouts_path = Path("eval/evaluation_set/consistency")
cons_renders_path = cons_layouts_path / "renders"
cons_jsons_path = cons_layouts_path / "json"


def get_relations(relation):
    if relation == "r":
        b1_pos = "left"
        b2_pos = "right"
    elif relation == "l":
        b1_pos = "right"
        b2_pos = "left"
    elif relation == "a":
        b1_pos = "bottom"
        b2_pos = "top"
    return b1_pos, b2_pos


def check_overlap(mask_1, mask_2):
    return (mask_1.bool() & mask_2.bool()).any()


def check_out_of_bounds(mask):
    xx, yy = np.where(mask.cpu())
    return xx.min() == 0 or xx.max() == 511 or yy.min() == 0 or yy.max() == 511


OBJECTS_CATEGORIES = {
    "animals": ["a cat", "a dog", "a horse", "an elephant", "a grizzly bear"],
    "indoor": ["a teddy bear", "a microwave", "a backpack", "an lcd tv", "a sofa", "a chair", "a table", "a bed"],
    "outdoor": ["a car", "a motorcycle", "a backpack", "a bench", "a sofa"],
}

# Width, Depth, Height
ASPECT_RATIOS = {
    "a cat": (0.2, 0.2, 0.4),
    "a dog": (0.3, 0.2, 0.5),
    "a horse": (0.7, 0.3, 0.5),
    "an elephant": (0.9, 0.3, 0.6),
    "a grizzly bear": (0.8, 0.3, 0.5),
    "a teddy bear": (0.3, 0.2, 0.4),
    "a microwave": (0.4, 0.2, 0.25),
    "a backpack": (0.3, 0.1, 0.4),
    "a car": (0.6, 1.2, 0.4),
    "a motorcycle": (0.8, 0.2, 0.35),
    "an lcd tv": (0.5, 0.05, 0.3),
    "a sofa": (0.9, 0.3, 0.4),
    "a chair": (0.25, 0.25, 0.5),
    "a table": (0.5, 0.5, 0.4),
    "a bench": (0.7, 0.3, 0.3),
    "a bed": (0.6, 0.7, 0.25),
}

RELATIONS = {
    "a cat": ("l", "r", "a"),
    "a dog": ("l", "r", "a"),
    "a horse": ("l", "r"),
    "an elephant": ("l", "r"),
    "a grizzly bear": ("l", "r"),
    "a teddy bear": ("l", "r", "a"),
    "a microwave": ("l", "r", "a"),
    "a backpack": ("l", "r", "a"),
    "a car": ("l", "r"),
    "a motorcycle": ("l", "r"),
    "an lcd tv": ("l", "r", "a"),
    "a sofa": ("l", "r"),
    "a chair": ("l", "r"),
    "a table": ("l", "r"),
    "a bench": ("l", "r"),
    "a bed": ("l", "r"),
}

SCENES = [
    ("An empty desert with cloudy sky", ["animals", "outdoor"]),
    ("An empty room with windows and curtains", ["indoor"]),
    ("An empty street", ["outdoor"]),
    ("An empty jungle", ["animals"]),
    ("An empty road", ["animals", "outdoor"]),
    ("An empty studio", ["indoor"]),
    ("An empty beach", ["animals"]),
    ("A snowy landscape", ["outdoor"]),
    ("An empty apartment", ["indoor"]),
]

YOLO8_LABELS = {
    "a cat": 15,
    "a dog": 16,
    "a horse": 17,
    "an elephant": 20,
    "a grizzly bear": 21,
    "a teddy bear": 77,
    "a microwave": 68,
    "a backpack": 24,
    "a car": 2,
    "a motorcycle": 3,
    "an lcd tv": 62,
    "a sofa": 57,
    "a chair": 56,
    "a table": 60,
    "a bench": 13,
    "a bed": 59,
}
