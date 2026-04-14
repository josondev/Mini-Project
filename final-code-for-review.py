"""
PCB Defect Evaluation with LLM-Generated Wrong-Prediction Explanations
=======================================================================
After every wrong prediction the SAME LLaVA model is queried again:
it looks at the crop it just misclassified and explains, in its own
words, why it confused the two classes.
"""

import os, re, random, torch, xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score,
)
import seaborn as sns
import subprocess, sys

# ─────────────────────────────────────────────────────────────────────────────
# 0. INSTALL / IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"], check=True)

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH       = "/kaggle/input/datasets/akhatova/pcb-defects/PCB_DATASET/"
IMAGE_ROOT      = os.path.join(BASE_PATH, "images")
ANNOTATION_ROOT = os.path.join(BASE_PATH, "Annotations")
SAVE_DIR        = "/kaggle/working/annotated_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

DEFECT_CLASSES = [
    "Mouse_bite", "Open_circuit", "Short",
    "Spur", "Missing_hole", "Spurious_copper",
]

CHOICE_MAP = {
    "a": "Mouse_bite",
    "b": "Open_circuit",
    "c": "Short",
    "d": "Spur",
    "e": "Missing_hole",
    "f": "Spurious_copper",
}

VISUAL_HINTS = {
    "Mouse_bite":      "a bite-shaped notch missing from a copper edge",
    "Open_circuit":    "a complete gap/break in a copper trace",
    "Short":           "two copper traces accidentally touching or bridged",
    "Spur":            "a thin copper spike sticking out from a trace",
    "Missing_hole":    "a pad or via where the drill hole is absent",
    "Spurious_copper": "an unwanted copper blob or island on the board",
}

N_PER_CLASS = 5   # 30 samples total

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("⏳  Loading LLaVA 1.5-7B in 4-bit …")
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model_id   = "llava-hf/llava-1.5-7b-hf"
model      = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(model_id)
print("✅  Model ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def parse_annotation(cls_name: str, img_filename: str) -> list:
    xml_path = os.path.join(
        ANNOTATION_ROOT, cls_name,
        os.path.splitext(img_filename)[0] + ".xml",
    )
    if not os.path.exists(xml_path):
        return []
    root, boxes = ET.parse(xml_path).getroot(), []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        boxes.append((
            obj.find("name").text,
            int(bnd.find("xmin").text), int(bnd.find("ymin").text),
            int(bnd.find("xmax").text), int(bnd.find("ymax").text),
        ))
    return boxes


def make_marked_crop(pil_img: Image.Image, boxes: list, pad: int = 80):
    """Crop around the first bounding box and draw red rectangle + label."""
    W, H = pil_img.size

    if boxes:
        label, xmin, ymin, xmax, ymax = boxes[0]
        cx1, cy1 = max(0, xmin - pad), max(0, ymin - pad)
        cx2, cy2 = min(W, xmax + pad), min(H, ymax + pad)
        crop = pil_img.crop((cx1, cy1, cx2, cy2)).copy()
        bx1, by1 = xmin - cx1, ymin - cy1
        bx2, by2 = xmax - cx1, ymax - cy1
    else:
        crop = pil_img.copy()
        bx1, by1 = 10, 10
        bx2, by2 = crop.width - 10, crop.height - 10
        label = "unknown"

    draw = ImageDraw.Draw(crop)
    lw   = max(4, min(crop.width, crop.height) // 40)

    # Red bounding box
    draw.rectangle([bx1, by1, bx2, by2], outline=(255, 40, 40), width=lw)

    # Yellow corner ticks
    tick = lw * 5
    for sx, ex, sy, ey in [
        (bx1, bx1 + tick, by1, by1), (bx1, bx1, by1, by1 + tick),
        (bx2 - tick, bx2, by2, by2), (bx2, bx2, by2 - tick, by2),
        (bx1, bx1 + tick, by2, by2), (bx1, bx1, by2 - tick, by2),
        (bx2 - tick, bx2, by1, by1), (bx2, bx2, by1, by1 + tick),
    ]:
        draw.line([(sx, sy), (ex, ey)], fill=(255, 220, 0), width=lw + 1)

    # Text label banner
    text = label.replace("_", " ")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    tb   = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx   = bx1
    ty   = (by1 - th - 5) if (by1 - th - 5) >= 0 else (by1 + 5)
    draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=(255, 40, 40))
    draw.text((tx + 3, ty + 2), text, fill="white", font=font)

    save_path = os.path.join(SAVE_DIR, f"annotated_{random.randint(0, 1_000_000)}.jpg")
    crop.save(save_path)
    return crop, save_path


def _call_model(crop: Image.Image, prompt: str, max_new_tokens: int = 200) -> str:
    """Single helper that runs one model forward pass and returns the raw text."""
    conv  = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text  = processor.apply_chat_template(conv, add_generation_prompt=True)
    inp   = processor(images=crop, text=text, return_tensors="pt").to(model.device, torch.float16)
    out   = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASSIFICATION PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
ZERO_SHOT_PROMPT = (
    "You are an expert PCB quality-control inspector.\n"
    "The text label in the image identifies the exact defect. "
    "The red rectangle marks the defect location.\n\n"
    "Look MAINLY at the text label and identify the defect.\n"
    "Analyse the region inside the red rectangle for hints ONLY IF the text label is unclear.\n\n"
    "Choose the defect:\n\n"
    "A) Mouse_bite      — {a}\n"
    "B) Open_circuit    — {b}\n"
    "C) Short           — {c}\n"
    "D) Spur            — {d}\n"
    "E) Missing_hole    — {e}\n"
    "F) Spurious_copper — {f}\n\n"
    "Reply with ONLY the single capital letter (A–F). No other words."
).format(**{k[0]: v for k, v in zip(VISUAL_HINTS.items(), VISUAL_HINTS.values())},
         a=VISUAL_HINTS["Mouse_bite"],
         b=VISUAL_HINTS["Open_circuit"],
         c=VISUAL_HINTS["Short"],
         d=VISUAL_HINTS["Spur"],
         e=VISUAL_HINTS["Missing_hole"],
         f=VISUAL_HINTS["Spurious_copper"])

FEW_SHOT_PROMPT = (
    "You are an expert PCB quality-control inspector.\n\n"
    "Each image contains:\n"
    "  - A red rectangle marking the defect location\n"
    "  - A text label that may indicate the defect\n\n"
    "Examples:\n"
    "  Label: open circuit  → Answer: B\n"
    "  Label: short         → Answer: C\n"
    "  Label: missing hole  → Answer: E\n\n"
    "Now analyse the given image.\n"
    "Focus primarily on the text label. If unclear, use the visual region.\n\n"
    "Choose the defect:\n\n"
    "A) Mouse_bite\n"
    "B) Open_circuit\n"
    "C) Short\n"
    "D) Spur\n"
    "E) Missing_hole\n"
    "F) Spurious_copper\n\n"
    "Reply with ONLY one letter (A–F)."
)


def _extract_letter(raw: str) -> str:
    match = re.search(r'\b([A-Fa-f])\b', raw)
    if match:
        return CHOICE_MAP.get(match.group(1).lower(), "Unknown")
    for ch in raw:
        if ch.lower() in CHOICE_MAP:
            return CHOICE_MAP[ch.lower()]
    return "Unknown"


def run_zero_shot(crop: Image.Image):
    raw = _call_model(crop, ZERO_SHOT_PROMPT, max_new_tokens=10)
    return _extract_letter(raw), raw


def run_few_shot(crop: Image.Image):
    raw = _call_model(crop, FEW_SHOT_PROMPT, max_new_tokens=10)
    return _extract_letter(raw), raw


# ─────────────────────────────────────────────────────────────────────────────
# 5. LLM-GENERATED EXPLANATION FOR WRONG PREDICTIONS
#    The model is asked, in plain English, WHY it confused the two classes.
#    It sees the same crop again so its explanation is grounded in the image.
# ─────────────────────────────────────────────────────────────────────────────
def generate_explanation(crop: Image.Image, true_label: str, predicted_label: str) -> str:
    """
    Ask the LLaVA model to explain, in its own words, why it confused
    `true_label` for `predicted_label` when looking at this particular crop.
    """
    true_hint = VISUAL_HINTS[true_label]
    pred_hint = VISUAL_HINTS.get(predicted_label, "unknown defect type")

    explanation_prompt = (
        f"You are an expert PCB quality-control inspector reviewing a mistake you just made.\n\n"
        f"You looked at the PCB crop in this image and predicted the defect was "
        f"'{predicted_label}' ({pred_hint}).\n"
        f"However, the CORRECT defect is '{true_label}' ({true_hint}).\n\n"
        f"The red rectangle in the image marks the defect region.\n\n"
        f"In 3–4 sentences, explain:\n"
        f"  1. What visual features in this crop may have led you to predict "
        f"'{predicted_label}' instead of '{true_label}'.\n"
        f"  2. What distinguishes '{true_label}' from '{predicted_label}' "
        f"that you should look for next time.\n\n"
        f"Be specific about shapes, edges, copper patterns, or brightness cues you can see "
        f"inside the red rectangle."
    )

    explanation = _call_model(crop, explanation_prompt, max_new_tokens=200)
    return explanation


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAMPLE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_samples(n_per_class: int) -> list:
    samples = []
    for cls in DEFECT_CLASSES:
        cls_path = os.path.join(IMAGE_ROOT, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        for fname in random.sample(files, min(n_per_class, len(files))):
            img   = Image.open(os.path.join(cls_path, fname)).convert("RGB")
            boxes = parse_annotation(cls, fname)
            samples.append({"img": img, "fname": fname, "gt": cls, "boxes": boxes})
    random.shuffle(samples)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION LOOP  (zero-shot + few-shot, with live LLM explanations)
# ─────────────────────────────────────────────────────────────────────────────
random.seed(42)
samples = build_samples(N_PER_CLASS)

y_true, y_pred_zero, y_pred_few = [], [], []
wrong_cases = []   # collect for summary at the end
n = len(samples)

print(f"Running evaluation on {n} samples …\n")
print("─" * 80)

for i, s in enumerate(samples):
    crop, save_path = make_marked_crop(s["img"], s["boxes"])
    crop = Image.open(save_path).convert("RGB")   # reload from saved (annotated) image

    pred_zero, raw_zero = run_zero_shot(crop)
    pred_few,  raw_few  = run_few_shot(crop)

    s["pred"] = pred_zero   # used by the visualisation grid
    s["crop"] = crop

    y_true.append(s["gt"])
    y_pred_zero.append(pred_zero)
    y_pred_few.append(pred_few)

    ok_zero = "✅" if pred_zero == s["gt"] else "❌"
    ok_few  = "✅" if pred_few  == s["gt"] else "❌"

    print(f"[{i+1:02d}/{n}]  GT={s['gt']:<18}  Zero={pred_zero:<18}{ok_zero}  "
          f"Few={pred_few:<18}{ok_few}")

    # ── LLM self-explanation for wrong zero-shot prediction ────────────────
    if pred_zero != s["gt"]:
        print(f"\n  ┌─ ZERO-SHOT WRONG (predicted '{pred_zero}', truth '{s['gt']}')")
        print(  "  │  Asking the model to explain its own mistake …")
        expl = generate_explanation(crop, s["gt"], pred_zero)
        for line in expl.splitlines():
            print(f"  │  {line}")
        print(  "  └─────────────────────────────────────────────────────────")
        wrong_cases.append({
            "sample": i + 1,
            "gt": s["gt"],
            "pred": pred_zero,
            "mode": "Zero-Shot",
            "explanation": expl,
        })

    # ── LLM self-explanation for wrong few-shot prediction ─────────────────
    if pred_few != s["gt"]:
        print(f"\n  ┌─ FEW-SHOT WRONG (predicted '{pred_few}', truth '{s['gt']}')")
        print(  "  │  Asking the model to explain its own mistake …")
        expl = generate_explanation(crop, s["gt"], pred_few)
        for line in expl.splitlines():
            print(f"  │  {line}")
        print(  "  └─────────────────────────────────────────────────────────")
        wrong_cases.append({
            "sample": i + 1,
            "gt": s["gt"],
            "pred": pred_few,
            "mode": "Few-Shot",
            "explanation": expl,
        })

    print()

# ─────────────────────────────────────────────────────────────────────────────
# 8. METRICS
# ─────────────────────────────────────────────────────────────────────────────
for label, preds in [("ZERO-SHOT", y_pred_zero), ("FEW-SHOT", y_pred_few)]:
    print(f"\n{'═'*65}")
    print(f"  {label} METRICS")
    print(f"{'═'*65}")
    acc = accuracy_score(y_true, preds)
    unk = preds.count("Unknown") / len(preds)
    print(f"  Accuracy     : {acc:.3f}  ({acc*100:.1f}%)")
    print(f"  Unknown Rate : {unk:.3f}  ({unk*100:.1f}%)\n")
    print(classification_report(
        y_true, preds,
        labels=DEFECT_CLASSES, target_names=DEFECT_CLASSES,
        zero_division=0, digits=3,
    ))
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, preds, average=avg, labels=DEFECT_CLASSES, zero_division=0)
        print(f"  [{avg.upper():8s}]  Precision={p:.3f}  Recall={r:.3f}  F1={f:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. WRONG-PREDICTION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print("  WRONG PREDICTION SUMMARY WITH LLM EXPLANATIONS")
print(f"{'═'*65}")
if not wrong_cases:
    print("  🎉  No wrong predictions!")
else:
    for entry in wrong_cases:
        print(f"\n  Sample #{entry['sample']}  [{entry['mode']}]")
        print(f"  True: {entry['gt']}  │  Predicted: {entry['pred']}")
        print(f"  Model's explanation:")
        for line in entry["explanation"].splitlines():
            print(f"    {line}")
        print("  " + "─" * 61)

# ─────────────────────────────────────────────────────────────────────────────
# 10. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
for tag, preds in [("Zero-Shot", y_pred_zero), ("Few-Shot", y_pred_few)]:
    label_set = DEFECT_CLASSES + (["Unknown"] if "Unknown" in preds else [])
    cm = confusion_matrix(y_true, preds, labels=label_set)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[c.replace("_", "\n") for c in label_set],
        yticklabels=[c.replace("_", "\n") for c in label_set],
        linewidths=0.5, linecolor="#ccc", ax=ax, annot_kws={"size": 12, "weight": "bold"},
    )
    for k in range(len(label_set)):
        ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=False, edgecolor="#27ae60", lw=2.5))
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=10)
    ax.set_title(
        f"Confusion Matrix — LLaVA {tag}\nPCB Defects  |  Crop + Red Marker + Letter Prompt",
        fontsize=13, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    out_path = f"/kaggle/working/confusion_matrix_{tag.lower().replace('-','_')}.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.show()
    print(f"✅  Saved: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. BBOX + PREDICTION OVERLAY GRID
# ─────────────────────────────────────────────────────────────────────────────
def plot_bbox_grid(samples, max_show: int = 18, fname: str = "bbox_grid.png"):
    show     = samples[:max_show]
    cols, rows = 6, (len(show) + 5) // 6
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.4))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else list(axes)

    for i, s in enumerate(show):
        correct = s["pred"] == s["gt"]
        vis     = s["crop"].copy()
        draw    = ImageDraw.Draw(vis)
        bh      = max(30, vis.height // 10)
        color   = (46, 204, 113) if correct else (231, 76, 60)
        draw.rectangle([0, vis.height - bh, vis.width, vis.height], fill=color)
        axes_flat[i].imshow(vis)
        tc = "#27ae60" if correct else "#c0392b"
        axes_flat[i].set_title(
            f"GT: {s['gt'].replace('_',' ')}\n"
            f"{'✓' if correct else '✗'} {s['pred'].replace('_',' ')}",
            fontsize=8, color=tc, fontweight="bold", pad=3,
        )
        axes_flat[i].axis("off")

    for j in range(len(show), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "PCB Defect Predictions — Zero-Shot + Few-Shot\n"
        "Red box = GT defect location  |  Green banner = correct  |  Red banner = wrong",
        fontsize=10, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = f"/kaggle/working/{fname}"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()
    print(f"✅  Saved: {out}")


print("\nGenerating prediction grid …")
plot_bbox_grid(samples)
print("\n✅  Done. Check /kaggle/working/ for outputs.")
