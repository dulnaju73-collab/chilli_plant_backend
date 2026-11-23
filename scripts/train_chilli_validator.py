"""Train, evaluate, and export the plant & chilli leaf vs. non-leaf TFLite validator.

Outputs a compact FP16-weighted TFLite classifier (FLOAT32 input/output):
  - Input:  [1, 160, 160, 3] float32 RGB normalized to [0, 1] (IMAGE_VALIDATION_INPUT_SCALE=0_1)
  - Output: [1, 2] float32 softmax probabilities [P(not_leaf), P(leaf)]
            (IMAGE_VALIDATION_POSITIVE_CLASS_INDEX=1, IMAGE_VALIDATION_INVERT_SCORE=false)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMG_SIZE = 160
SEED = 42

CALTECH_EXCLUDED_LABELS = {
    "bonsai",
    "joshua_tree",
    "lotus",
    "strawberry",
    "sunflower",
    "water_lilly",
}

TINY_IMAGENET_EXCLUDED_LABELS = {
    "n02226429",  # grasshopper (often photographed on green leaves)
    "n07715103",  # cauliflower
    "n07720875",  # bell pepper
    "n07749582",  # lemon
    "n07753592",  # banana
    "n07768694",  # pomegranate
    "n12267677",  # acorn
    "n13001041",  # mushroom
    "n14991210",  # orange
}


def _to_resized_rgb(img: Any) -> Image.Image | None:
    try:
        if isinstance(img, dict) and img.get("bytes"):
            with Image.open(io.BytesIO(img["bytes"])) as raw:
                if raw.width < 32 or raw.height < 32:
                    return None
                return raw.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        if isinstance(img, Image.Image):
            if img.width < 32 or img.height < 32:
                return None
            return img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        return None
    except Exception:
        return None


def _save_rgb_jpeg(img: Any, dest: Path) -> bool:
    try:
        pil = img if isinstance(img, Image.Image) else _to_resized_rgb(img)
        if pil is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        pil.save(dest, format="JPEG", quality=92)
        return True
    except Exception:
        return False


def _save_split_items(
    items: list[tuple[Any, str]],
    cache_dir: Path,
    class_dir: str,
    prefix: str,
    n_train: int,
    n_val: int,
    n_test: int,
    rng: random.Random,
    *,
    group_split: bool = False,
) -> dict[str, int]:
    """Split items into train/val/test either stratified by label or grouped by source photo."""
    by_group: dict[str, list[Any]] = defaultdict(list)
    for img, grp in items:
        by_group[grp].append(img)

    groups = list(by_group.keys())
    rng.shuffle(groups)
    targets = {"train": n_train, "val": n_val, "test": n_test}
    active = [name for name, cap in targets.items() if cap > 0]
    total_target = sum(targets[name] for name in active)

    split_buckets: dict[str, list[Any]] = {"train": [], "val": [], "test": []}
    if group_split and len(groups) >= len(active):
        # Group-level split so crops from the same source_photo never cross splits
        for grp in groups:
            g_imgs = by_group[grp]
            candidates = [name for name in active if len(split_buckets[name]) < targets[name]]
            if not candidates:
                break
            chosen = min(candidates, key=lambda name: len(split_buckets[name]) / targets[name])
            split_buckets[chosen].extend(g_imgs)
    else:
        # Stratified split within each class label
        train_ratio = n_train / total_target
        val_ratio = n_val / total_target
        for grp in groups:
            g_imgs = list(by_group[grp])
            rng.shuffle(g_imgs)
            n = len(g_imgs)
            i_tr = int(round(n * train_ratio))
            i_va = int(round(n * (train_ratio + val_ratio)))
            if n_train > 0:
                split_buckets["train"].extend(g_imgs[:i_tr])
            if n_val > 0:
                split_buckets["val"].extend(g_imgs[i_tr:i_va])
            if n_test > 0:
                split_buckets["test"].extend(g_imgs[i_va:])

    counts = {"train": 0, "val": 0, "test": 0}
    for split, limit in targets.items():
        if limit <= 0:
            continue
        rng.shuffle(split_buckets[split])
        for idx, img in enumerate(split_buckets[split][:limit]):
            dest = cache_dir / split / class_dir / f"{prefix}_{idx:05d}.jpg"
            if _save_rgb_jpeg(img, dest):
                counts[split] += 1
    return counts


def _stream_balanced(
    ds_name: str,
    split: str,
    label_col: str,
    max_total: int,
    max_per_class: int | None = None,
    exclude_substrings: tuple[str, ...] = (),
    exclude_exact: set[str] | None = None,
    group_col: str | None = None,
    max_scan: int = 50000,
    config_name: str | None = None,
) -> list[tuple[Image.Image, str]]:
    from datasets import load_dataset

    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            ds = load_dataset(ds_name, name=config_name, split=split, streaming=True)
            features = ds.features
            label_names = None
            if label_col in features and hasattr(features[label_col], "names"):
                label_names = features[label_col].names

            effective_max_per_class = max_per_class
            if effective_max_per_class is None:
                if label_names:
                    allowed = [
                        n
                        for n in label_names
                        if not any(sub in n.lower() for sub in exclude_substrings)
                        and (not exclude_exact or n.lower() not in exclude_exact)
                    ]
                    effective_max_per_class = max(2, (max_total // max(1, len(allowed))) + 2)
                else:
                    effective_max_per_class = max_total

            ds = ds.shuffle(seed=SEED, buffer_size=64)
            per_class: dict[str, int] = defaultdict(int)
            collected: list[tuple[Image.Image, str]] = []
            scanned = 0

            for row in ds:
                scanned += 1
                if scanned > max_scan or len(collected) >= max_total:
                    break
                raw_lbl = row.get(label_col, 0)
                lbl_name = (
                    label_names[raw_lbl]
                    if (label_names is not None and isinstance(raw_lbl, int) and 0 <= raw_lbl < len(label_names))
                    else str(raw_lbl)
                )
                low = lbl_name.lower()
                if any(sub in low for sub in exclude_substrings):
                    continue
                if exclude_exact and low in exclude_exact:
                    continue
                if per_class[lbl_name] >= effective_max_per_class:
                    continue
                resized = _to_resized_rgb(row.get("image"))
                if resized is None:
                    continue
                grp = str(row.get(group_col)) if group_col and row.get(group_col) is not None else lbl_name
                per_class[lbl_name] += 1
                collected.append((resized, grp))

            print(
                f"  [{ds_name}:{split}] collected {len(collected)} images across {len(per_class)} classes "
                f"(cap/class={effective_max_per_class}, scanned={scanned})",
                flush=True,
            )
            return collected
        except Exception as exc:
            last_exc = exc
            print(f"  [{ds_name}:{split}] attempt {attempt} failed: {exc}", flush=True)
            time.sleep(attempt * 5)
    raise RuntimeError(f"Failed to stream {ds_name}:{split}") from last_exc


def _generate_synthetic_negatives(cache_dir: Path, rng: random.Random) -> dict[str, int]:
    """Generate solid colors (incl. greens), gradients, noise, and geometric patterns as non-leaf negatives."""
    np_rng = np.random.default_rng(SEED)
    counts = {"train": 0, "val": 0, "test": 0}
    specs = [("train", 150), ("val", 40), ("test", 50)]

    for split, n in specs:
        for i in range(n):
            mode = i % 5
            if mode == 0:
                # Solid green / foliage-colored rectangles (hardest naive color trap!)
                r = int(np_rng.integers(0, 90))
                g = int(np_rng.integers(90, 256))
                b = int(np_rng.integers(0, 90))
                arr = np.full((IMG_SIZE, IMG_SIZE, 3), (r, g, b), dtype=np.uint8)
            elif mode == 1:
                # Arbitrary solid colors (white, black, gray, brown, red, blue)
                color = np_rng.integers(0, 256, size=3, dtype=np.uint8)
                arr = np.broadcast_to(color, (IMG_SIZE, IMG_SIZE, 3)).copy()
            elif mode == 2:
                # Random uniform noise
                arr = np_rng.integers(0, 256, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            elif mode == 3:
                # Smooth gradients (including green gradients)
                c1 = np_rng.integers(0, 256, size=(1, 1, 3)).astype(np.float32)
                c2 = np_rng.integers(0, 256, size=(1, 1, 3)).astype(np.float32)
                t = np.linspace(0.0, 1.0, IMG_SIZE, dtype=np.float32)
                t = t[:, None, None] if (i % 2 == 0) else t[None, :, None]
                arr = np.clip(c1 * (1.0 - t) + c2 * t, 0, 255).astype(np.uint8)
                arr = np.broadcast_to(arr, (IMG_SIZE, IMG_SIZE, 3)).copy()
            else:
                # Checkerboard / stripes
                step = int(np_rng.integers(8, 32))
                c1 = np_rng.integers(0, 256, size=3, dtype=np.uint8)
                c2 = np_rng.integers(0, 256, size=3, dtype=np.uint8)
                yy, xx = np.indices((IMG_SIZE, IMG_SIZE))
                mask = ((yy // step) + (xx // step)) % 2 == 0
                arr = np.where(mask[:, :, None], c1, c2).astype(np.uint8)
            dest = cache_dir / split / "0_not_leaf" / f"synth_{i:05d}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr, mode="RGB").save(dest, format="JPEG", quality=92)
            counts[split] += 1
    return counts


def prepare_dataset(cache_dir: Path) -> None:
    marker = cache_dir / ".prepared_v2_leaf.json"
    if marker.exists():
        print(f"Using cached dataset at {cache_dir} ({marker.read_text(encoding='utf-8')})", flush=True)
        return

    rng = random.Random(SEED)
    summary: dict[str, dict[str, int]] = {}
    print("Downloading and caching balanced leaf (incl. chilli) + non-leaf images...", flush=True)

    # --- POSITIVES (class 1: plant & chilli leaves across field, phone, Google, and lab domains) ---
    cpv = _stream_balanced(
        "Project-AgML/chili_pepper_virus_classification",
        "train",
        "label",
        max_total=1050,
        max_per_class=400,
        group_col="source_photo",
        max_scan=4200,
    )
    summary["pos_cpv"] = _save_split_items(cpv, cache_dir, "1_leaf", "cpv", 700, 150, 150, rng, group_split=True)

    waruni = _stream_balanced(
        "WaruniSHA/chilli-diseases",
        "train",
        "label",
        max_total=950,
        exclude_exact={"others"},
        max_scan=1200,
    )
    summary["pos_waruni"] = _save_split_items(waruni, cache_dir, "1_leaf", "waruni", 650, 150, 150, rng)

    plantdoc = _stream_balanced(
        "Project-AgML/plant_doc_classification",
        "train",
        "label",
        max_total=660,
        max_scan=3000,
    )
    summary["pos_plantdoc"] = _save_split_items(plantdoc, cache_dir, "1_leaf", "plantdoc", 450, 100, 100, rng)

    for split_src, split_dst, limit in (("train", "train", 380), ("validation", "val", 100), ("test", "test", 100)):
        beans = _stream_balanced("AI-Lab-Makerere/beans", split_src, "labels", max_total=limit, max_scan=1500)
        cnt = 0
        for idx, (img, _) in enumerate(beans):
            if _save_rgb_jpeg(img, cache_dir / split_dst / "1_leaf" / f"beans_{idx:05d}.jpg"):
                cnt += 1
        summary.setdefault("pos_beans", {})[split_dst] = cnt

    pv_tr = _stream_balanced(
        "BrandonFors/Plant-Diseases-PlantVillage-Dataset",
        "train",
        "label",
        max_total=490,
        max_scan=45000,
    )
    summary["pos_pv_tr"] = _save_split_items(pv_tr, cache_dir, "1_leaf", "plantvillage", 400, 80, 0, rng)
    pv_te = _stream_balanced(
        "BrandonFors/Plant-Diseases-PlantVillage-Dataset",
        "test",
        "label",
        max_total=100,
        max_scan=12000,
    )
    cnt = 0
    for idx, (img, _) in enumerate(pv_te):
        if _save_rgb_jpeg(img, cache_dir / "test" / "1_leaf" / f"plantvillage_{idx:05d}.jpg"):
            cnt += 1
    summary["pos_pv_te"] = {"test": cnt}

    tea = _stream_balanced("Project-AgML/tea_leaf_disease_classification", "train", "label", max_total=470, max_scan=6000)
    summary["pos_tea"] = _save_split_items(tea, cache_dir, "1_leaf", "tea", 320, 70, 70, rng)

    # 100% ZERO-SHOT EXTERNAL CHILLI LEAF TEST SET (never seen during training or validation!)
    cold_zeroshot = _stream_balanced(
        "Project-AgML/COLD_chili_leaf_disease_classification",
        "train",
        "label",
        max_total=532,
        max_per_class=200,
        max_scan=600,
        config_name="raw",
    )
    cnt = 0
    for idx, (img, _) in enumerate(cold_zeroshot):
        if _save_rgb_jpeg(img, cache_dir / "test" / "1_leaf" / f"cold_{idx:05d}.jpg"):
            cnt += 1
    summary["pos_cold_zeroshot"] = {"test": cnt}

    # --- NEGATIVES (class 0: non-leaf / non-plant everyday photos + synthetic traps) ---
    pets_tr = _stream_balanced("timm/oxford-iiit-pet", "train", "label", max_total=800, max_scan=5000)
    summary["neg_pets_tr"] = _save_split_items(pets_tr, cache_dir, "0_not_leaf", "pets", 650, 140, 0, rng)
    pets_te = _stream_balanced("timm/oxford-iiit-pet", "test", "label", max_total=200, max_scan=4000)
    cnt = 0
    for idx, (img, _) in enumerate(pets_te):
        if _save_rgb_jpeg(img, cache_dir / "test" / "0_not_leaf" / f"pets_{idx:05d}.jpg"):
            cnt += 1
    summary["neg_pets_te"] = {"test": cnt}

    caltech = _stream_balanced(
        "flwrlabs/caltech101",
        "train",
        "label",
        max_total=1020,
        exclude_exact=CALTECH_EXCLUDED_LABELS,
        max_scan=9500,
    )
    summary["neg_caltech"] = _save_split_items(caltech, cache_dir, "0_not_leaf", "caltech", 700, 150, 160, rng)

    imagenette = _stream_balanced("johnowhitaker/imagenette2-320", "train", "label", max_total=1020, max_scan=14000)
    summary["neg_imagenette"] = _save_split_items(imagenette, cache_dir, "0_not_leaf", "imagenette", 700, 150, 160, rng)

    tiny_tr = _stream_balanced(
        "zh-plus/tiny-imagenet",
        "train",
        "label",
        max_total=1040,
        exclude_exact=TINY_IMAGENET_EXCLUDED_LABELS,
        max_scan=100000,
    )
    summary["neg_tiny_tr"] = _save_split_items(tiny_tr, cache_dir, "0_not_leaf", "tinyimg", 850, 180, 0, rng)
    tiny_te = _stream_balanced(
        "zh-plus/tiny-imagenet",
        "valid",
        "label",
        max_total=180,
        exclude_exact=TINY_IMAGENET_EXCLUDED_LABELS,
        max_scan=10000,
    )
    cnt = 0
    for idx, (img, _) in enumerate(tiny_te):
        if _save_rgb_jpeg(img, cache_dir / "test" / "0_not_leaf" / f"tinyimg_{idx:05d}.jpg"):
            cnt += 1
    summary["neg_tiny_te"] = {"test": cnt}

    summary["neg_synth"] = _generate_synthetic_negatives(cache_dir, rng)

    marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Dataset preparation summary:", json.dumps(summary, indent=2), flush=True)


def _load_split_arrays(split_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths_and_labels: list[tuple[Path, int]] = []
    for cls_idx, cls_name in ((0, "0_not_leaf"), (1, "1_leaf")):
        folder = split_dir / cls_name
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.jpg")):
            paths_and_labels.append((p, cls_idx))
    rng = random.Random(SEED)
    rng.shuffle(paths_and_labels)

    xs = np.empty((len(paths_and_labels), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    ys = np.empty((len(paths_and_labels),), dtype=np.int32)
    names: list[str] = []
    for i, (p, lbl) in enumerate(paths_and_labels):
        with Image.open(p) as im:
            xs[i] = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        ys[i] = lbl
        names.append(p.name)
    return xs, ys, names


def train_and_export(cache_dir: Path, output_path: Path, report_path: Path | None) -> None:
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score

    tf.keras.utils.set_random_seed(SEED)
    print("Loading cached train/val/test tensors...", flush=True)
    x_train, y_train, _ = _load_split_arrays(cache_dir / "train")
    x_val, y_val, _ = _load_split_arrays(cache_dir / "val")
    x_test, y_test, test_names = _load_split_arrays(cache_dir / "test")

    print(
        f"Loaded tensors: train={len(x_train)} (pos={int(y_train.sum())}, neg={int((y_train==0).sum())}), "
        f"val={len(x_val)} (pos={int(y_val.sum())}, neg={int((y_val==0).sum())}), "
        f"test={len(x_test)} (pos={int(y_test.sum())}, neg={int((y_test==0).sum())})",
        flush=True,
    )

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=2)

    def augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        # Random zoom/crop (65% - 100%) to handle varied phone framing
        crop_frac = tf.random.uniform([], minval=0.65, maxval=1.0)
        crop_size = tf.cast(tf.round(crop_frac * IMG_SIZE), tf.int32)
        image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method="bilinear")
        image = tf.image.random_brightness(image, max_delta=0.20)
        image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
        image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
        image = tf.image.random_hue(image, max_delta=0.03)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    batch_size = 32
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train_oh))
        .shuffle(len(x_train), seed=SEED, reshuffle_each_iteration=True)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val_oh)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Clean inference model expecting [0, 1] float32 RGB inputs -> [not_chilli, chilli_leaf] softmax
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype="float32", name="rgb_0_1")
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="to_mobilenet_range")(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=0.75,
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="chilli_probs")(x)
    model = tf.keras.Model(inputs, outputs, name="chilli_leaf_validator")

    # Balanced class weights (avoid biasing against real-world leaf photos)
    n_neg = float((y_train == 0).sum())
    n_pos = float((y_train == 1).sum())
    total = n_neg + n_pos
    class_weight = {0: total / (2.0 * n_neg), 1: total / (2.0 * n_pos)}
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03)

    print("Phase 1: Training classification head (frozen MobileNetV2 alpha=0.75)...", flush=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn, metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=5, class_weight=class_weight, verbose=2)

    print("Phase 2: Fine-tuning top backbone layers...", flush=True)
    backbone.trainable = True
    for layer in backbone.layers[:-45]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1.5e-4), loss=loss_fn, metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=10, class_weight=class_weight, callbacks=callbacks, verbose=2)

    # Export via Keras 3 SavedModel -> TFLiteConverter.from_saved_model so all weights are frozen
    # (avoids unfrozen READ_VARIABLE ops from from_concrete_functions under Keras 3) and locks
    # static input shape [1, 160, 160, 3] FLOAT32 -> [1, 2] FLOAT32.
    print(f"Exporting TFLite model to {output_path}...", flush=True)
    export_input = tf.keras.Input(batch_shape=(1, IMG_SIZE, IMG_SIZE, 3), dtype="float32", name="rgb_0_1")
    export_model = tf.keras.Model(export_input, model(export_input, training=False), name="chilli_leaf_validator_export")
    saved_model_dir = cache_dir / "saved_model_export"
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    export_model.export(str(saved_model_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()
    if len(tflite_bytes) < 1_000_000:
        raise RuntimeError(f"Exported TFLite model is unexpectedly small ({len(tflite_bytes)} bytes); weights may not be frozen.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_bytes)
    sha256 = hashlib.sha256(tflite_bytes).hexdigest()
    print(f"Exported {output_path} ({len(tflite_bytes)/1e6:.2f} MB, sha256={sha256})", flush=True)

    # End-to-end evaluation through production ChilliImageValidator
    from app.models.chilli_validator import ChilliImageValidator

    validator = ChilliImageValidator(output_path, positive_class_index=1, threshold=0.55, invert_score=False, input_scale="0_1")
    scores = np.empty(len(x_test), dtype=np.float32)
    for i, name in enumerate(test_names):
        cls_dir = "1_leaf" if y_test[i] == 1 else "0_not_leaf"
        with Image.open(cache_dir / "test" / cls_dir / name) as im:
            _, score = validator.validate(im.convert("RGB"))
            scores[i] = score

    auc = float(roc_auc_score(y_test, scores))
    print(f"\n=== TFLite Held-Out Test Evaluation (n={len(y_test)}, ROC-AUC={auc:.4f}) ===", flush=True)

    threshold_metrics = {}
    for thr in (0.45, 0.50, 0.55, 0.65, 0.75):
        preds = (scores >= thr).astype(np.int32)
        tp = int(((preds == 1) & (y_test == 1)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        tn = int(((preds == 0) & (y_test == 0)).sum())
        recall = tp / max(1, tp + fn)
        far = fp / max(1, fp + tn)
        precision = tp / max(1, tp + fp)
        acc = (tp + tn) / max(1, len(y_test))
        threshold_metrics[f"{thr:.2f}"] = dict(accuracy=round(acc, 4), recall=round(recall, 4), precision=round(precision, 4), false_accept_rate=round(far, 4), tp=tp, fn=fn, fp=fp, tn=tn)
        print(f"  thr={thr:.2f} | acc={acc:.4f} | leaf_recall={recall:.4f} ({tp}/{tp+fn}) | precision={precision:.4f} | non_leaf_FAR={far:.4f} ({fp}/{fp+tn})", flush=True)

    print("\n=== Per-Source Breakdown at threshold=0.55 (POS:cold is 100% zero-shot unseen chilli leaves) ===", flush=True)
    by_src: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for name, lbl, sc in zip(test_names, y_test, scores):
        src = name.split("_")[0]
        by_src[f"{'POS' if lbl == 1 else 'NEG'}:{src}"].append((int(lbl), float(sc)))

    source_metrics = {}
    for key in sorted(by_src):
        rows = by_src[key]
        lbl = rows[0][0]
        sc_arr = np.array([r[1] for r in rows])
        accepted = int((sc_arr >= 0.55).sum())
        rate = accepted / len(rows)
        status_label = "recall" if lbl == 1 else "false_accept"
        source_metrics[key] = dict(count=len(rows), accepted_at_055=accepted, rate=round(rate, 4), mean_score=round(float(sc_arr.mean()), 4))
        print(f"  {key:20s} | n={len(rows):3d} | {status_label}={rate*100:5.1f}% ({accepted}/{len(rows)}) | mean_p_leaf={sc_arr.mean():.3f}", flush=True)

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "model_path": str(output_path),
                    "size_bytes": len(tflite_bytes),
                    "sha256": sha256,
                    "input_shape": [1, IMG_SIZE, IMG_SIZE, 3],
                    "input_scale": "0_1",
                    "positive_class_index": 1,
                    "invert_score": False,
                    "recommended_threshold": 0.55,
                    "roc_auc": round(auc, 4),
                    "thresholds": threshold_metrics,
                    "per_source_at_0_55": source_metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Strict quality gate at production default threshold 0.55, including zero-shot unseen chilli recall
    m055 = threshold_metrics["0.55"]
    cold_zeroshot_recall = source_metrics.get("POS:cold", {}).get("rate", 0.0)
    pets_far = source_metrics.get("NEG:pets", {}).get("rate", 0.0)
    synth_far = source_metrics.get("NEG:synth", {}).get("rate", 0.0)
    failures: list[str] = []
    if auc < 0.98:
        failures.append(f"ROC-AUC {auc:.4f} < 0.98")
    if cold_zeroshot_recall < 0.92:
        failures.append(f"Zero-shot unseen chilli (COLD) recall {cold_zeroshot_recall:.4f} < 0.92 at threshold 0.55")
    if m055["recall"] < 0.94:
        failures.append(f"Overall leaf recall {m055['recall']:.4f} < 0.94 at threshold 0.55")
    if m055["false_accept_rate"] > 0.03:
        failures.append(f"Non-leaf FAR {m055['false_accept_rate']:.4f} > 0.03 at threshold 0.55")
    if pets_far > 0.02:
        failures.append(f"Dogs/pets FAR {pets_far:.4f} > 0.02 at threshold 0.55")
    if synth_far > 0.02:
        failures.append(f"Synthetic/solid FAR {synth_far:.4f} > 0.02 at threshold 0.55")
    if failures:
        raise SystemExit("Validator quality gate failed: " + "; ".join(failures))
    print("Validator quality gate PASSED.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the chilli leaf validator TFLite model.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory for cached resized JPEG splits.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models" / "leaf_detection.tflite")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON evaluation report path.")
    args = parser.parse_args()
    prepare_dataset(args.cache_dir)
    train_and_export(args.cache_dir, args.output, args.report)


if __name__ == "__main__":
    main()
