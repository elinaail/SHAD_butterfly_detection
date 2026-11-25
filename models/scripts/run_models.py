"""Run the three project models, keeping resumable predictions and provenance."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

MODEL_IDS = {
    "dino": "IDEA-Research/grounding-dino-tiny",
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "qwen": "Qwen/Qwen3-VL-2B-Instruct",
}
PROMPT = (
    "Locate every adult butterfly and moth in the image, including small moths. "
    "Do not include caterpillars, pupae, other insects, flowers or leaves. "
    "Return only a JSON array of boxes: [[x_min, y_min, x_max, y_max], ...]. "
    "Coordinates must be integers from 0 to 1000, relative to the whole image. "
    "Use tight bounding boxes covering the entire visible insect. Return [] if none are present. No labels or explanations."
)


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def box_from_xyxy(coords, width, height):
    values = [float(v) for v in coords]
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid bounding box")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Reversed or empty box")
    x1, x2 = [min(1.0, max(0.0, x / width)) for x in (x1, x2)]
    y1, y2 = [min(1.0, max(0.0, y / height)) for y in (y1, y2)]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box is outside the image")
    return dict(left=x1, top=y1, width=x2 - x1, height=y2 - y1)


def iou(a, b):
    w = max(
        0.0,
        min(a["left"] + a["width"], b["left"] + b["width"]) - max(a["left"], b["left"]),
    )
    h = max(
        0.0,
        min(a["top"] + a["height"], b["top"] + b["height"]) - max(a["top"], b["top"]),
    )
    intersection = w * h
    return intersection / (
        a["width"] * a["height"] + b["width"] * b["height"] - intersection
    )


def nms(boxes, scores, threshold=0.5):
    kept = []
    for index in sorted(range(len(scores)), key=lambda i: -scores[i]):
        if all(iou(boxes[index], boxes[j]) <= threshold for j in kept):
            kept.append(index)
    return kept


def parse_qwen(text):
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(clean)
    if not isinstance(result, list):
        raise ValueError("Expected a JSON array")
    return qwen_items_to_boxes(result)


def qwen_items_to_boxes(result):
    boxes = []
    for item in result:
        if isinstance(item, list):
            coords = item
        elif isinstance(item, dict):
            coords = item.get("box", item.get("bbox_2d"))
            if coords is None:
                raise ValueError("Missing bounding box coordinates")
        else:
            raise ValueError("Invalid bounding box item")
        boxes.append(box_from_xyxy(coords, 1000, 1000))
    return boxes


def parse_qwen_truncated(text):
    """Recover only complete top-level items at a verified generation token limit.

    No closing brackets, coordinates or labels are invented. Repeated boxes are
    preserved: they are actual model predictions, including any hallucinations.
    """
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].strip()
    if not clean.startswith("["):
        raise ValueError("Truncated answer does not start with a JSON array")
    decoder, items, position = json.JSONDecoder(), [], 1
    while position < len(clean):
        while position < len(clean) and clean[position].isspace():
            position += 1
        try:
            item, end = decoder.raw_decode(clean, position)
        except json.JSONDecodeError:
            break
        items.append(item)
        position = end
        while position < len(clean) and clean[position].isspace():
            position += 1
        if position >= len(clean) or clean[position] != ",":
            break
        position += 1
    if not items:
        raise ValueError("Truncated answer contains no complete boxes")
    return qwen_items_to_boxes(items)


def load_image(row, root):
    from PIL import Image
    import requests

    relative = Path(row["local_path"].replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe local_path")
    path = root / relative
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                response = requests.get(row["url"], timeout=90)
                response.raise_for_status()
                from io import BytesIO

                with Image.open(BytesIO(response.content)) as check:
                    check.verify()
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_bytes(response.content)
                tmp.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    # Match the original pixel orientation used by CVAT (no EXIF transpose).
    with Image.open(path) as image:
        result = image.convert("RGB")
    return result, hashlib.sha256(path.read_bytes()).hexdigest()


def sync(device):
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument(
        "--input", type=Path, default=Path("inaturalist_dataset/main.jsonl")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument(
        "--revision", help="Hugging Face commit; resolved commit is recorded"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--photo-ids",
        type=int,
        nargs="+",
        help="Optional photo IDs for a smoke test on the same manifest",
    )
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Process downloaded images; missing images still prevent completion",
    )
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Qwen prompt override; exact text is recorded in metadata",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--image-max-side", type=int, default=768)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Reject Qwen token-limit responses instead of recovering a prefix",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=["eager", "sdpa"],
        default="sdpa",
        help="Qwen attention backend",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Candidate floor; export higher thresholds later",
    )
    args = parser.parse_args()
    qwen_prompt = args.prompt_file.read_text().strip() if args.prompt_file else PROMPT
    if (
        args.max_new_tokens < 1
        or args.image_max_side < 32
        or args.repetition_penalty < 1
    ):
        parser.error("Invalid generation or image-size setting")
    import torch
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
    )

    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    torch.manual_seed(42)
    torch.set_num_threads(4)
    rows = read_jsonl(args.input)
    if len({r["photo_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate photo_id in input")
    if args.photo_ids:
        requested = set(args.photo_ids)
        if requested - {r["photo_id"] for r in rows}:
            raise ValueError("Requested photo_id is not in the manifest")
        rows = [r for r in rows if r["photo_id"] in requested]
    if args.limit:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Prevent two resumptions from appending duplicate photo IDs to the same run.
    import fcntl

    run_lock = args.output.with_suffix(".lock").open("w")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another process is writing this output")
    metadata_path = args.output.with_suffix(".meta.json")
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if previous and args.revision and args.revision != previous["resolved_revision"]:
        raise ValueError("A resumed run must use its recorded model commit")
    effective_revision = args.revision or (
        previous["resolved_revision"] if previous else None
    )
    config = dict(
        model=MODEL_IDS[args.model],
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        threshold=args.threshold,
        nms_iou=0.5,
        device=device,
        prompt=(
            qwen_prompt
            if args.model == "qwen"
            else (
                "butterfly. moth."
                if args.model == "dino"
                else ["a photo of a butterfly", "a photo of a moth"]
            )
        ),
        revision_requested=(
            previous["config"]["revision_requested"] if previous else args.revision
        ),
        max_new_tokens=args.max_new_tokens if args.model == "qwen" else None,
    )
    if args.model == "qwen":
        config["qwen_max_image_side"] = args.image_max_side
        config["qwen_processor_backend"] = "torchvision"
        config["attention_implementation"] = args.attention_implementation
        if args.repetition_penalty != 1.0:
            config["repetition_penalty"] = args.repetition_penalty
        if args.require_complete:
            config["require_complete"] = True
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old["config"] != config:
            raise ValueError(
                "Output belongs to another configuration; choose a new path"
            )
        for package, version in old["versions"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(
                    f"{package} version changed; resume in the original environment or choose a new output"
                )
    elif args.output.exists():
        raise ValueError(
            "Existing predictions have no metadata; choose a new output path"
        )
    done = (
        {r["photo_id"] for r in read_jsonl(args.output)}
        if args.output.exists()
        else set()
    )
    pending = [r for r in rows if r["photo_id"] not in done]
    if not pending:
        print("All requested images already complete", flush=True)
        return
    if args.cached_only:
        pending = [
            r
            for r in pending
            if (args.root / r["local_path"].replace("\\", "/")).exists()
        ]
        if not pending:
            raise SystemExit(
                "No remaining downloaded images; run without --cached-only after downloading"
            )
    started = time.perf_counter()
    from huggingface_hub import hf_hub_download

    revision = Path(
        hf_hub_download(
            MODEL_IDS[args.model], "config.json", revision=effective_revision
        )
    ).parent.name
    kwargs = {"revision": revision}
    processor = AutoProcessor.from_pretrained(
        MODEL_IDS[args.model],
        backend="torchvision" if args.model == "qwen" else "pil",
        **kwargs,
    )
    if args.model == "qwen":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_IDS[args.model],
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            attn_implementation=args.attention_implementation,
            **kwargs,
        )
    else:
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_IDS[args.model], **kwargs
        )
    model = model.to(device).eval()
    sync(device)
    load_seconds = time.perf_counter() - started
    if metadata_path.exists() and old["resolved_revision"] != revision:
        raise ValueError(
            "Model revision changed; pass --revision from previous metadata"
        )
    metadata = dict(
        config=config,
        resolved_revision=revision,
        load_seconds=load_seconds,
        python=platform.python_version(),
        platform=platform.platform(),
        versions={
            p: importlib.metadata.version(p)
            for p in (
                ["torch", "transformers", "Pillow", "torchvision"]
                if args.model == "qwen"
                else ["torch", "transformers", "Pillow"]
            )
        },
        timing="RGB image ready -> preprocessing + model + postprocessing + device synchronization; excludes download and model loading",
        truncated_response_policy=(
            "Reject any response without EOS; do not recover partial JSON."
            if args.require_complete
            else "At a verified max_new_tokens limit, retain only complete JSON array items and mark status=truncated. Preserve duplicate boxes and the full raw answer."
        ),
    )
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    consecutive_errors = 0
    for index, row in enumerate(pending):
        base = dict(
            photo_id=row["photo_id"], model=MODEL_IDS[args.model], schema_version=1
        )
        image = None
        sha256 = None
        inference_started = None
        ended_with_eos = False
        raw_text = None
        response_truncated = False
        generated_tokens = None
        try:
            image, sha256 = load_image(row, args.root)
            sync(device)
            started = time.perf_counter()
            inference_started = started
            raw_text = None
            with torch.inference_mode():
                if args.model == "qwen":
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image.resize(
                                        _size(image, args.image_max_side)
                                    ),
                                },
                                {"type": "text", "text": qwen_prompt},
                            ],
                        }
                    ]
                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(device)
                    output = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=args.repetition_penalty,
                    )
                    raw_text = processor.batch_decode(
                        output[:, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )[0]
                    generated_tokens = output.shape[1] - inputs["input_ids"].shape[1]
                    eos_ids = model.generation_config.eos_token_id
                    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
                    ended_with_eos = int(output[0, -1]) in eos_ids
                    if args.require_complete and not ended_with_eos:
                        raise ValueError("Qwen response did not finish with EOS")
                    try:
                        boxes = parse_qwen(raw_text)
                    except json.JSONDecodeError:
                        if (
                            args.require_complete
                            or generated_tokens < args.max_new_tokens
                            or ended_with_eos
                        ):
                            raise
                        boxes = parse_qwen_truncated(raw_text)
                        response_truncated = True
                    scores = [None] * len(boxes)
                else:
                    text = (
                        "butterfly. moth."
                        if args.model == "dino"
                        else [["a photo of a butterfly", "a photo of a moth"]]
                    )
                    inputs = processor(images=image, text=text, return_tensors="pt").to(
                        device
                    )
                    output = model(**inputs)
                    if args.model == "dino":
                        result = processor.post_process_grounded_object_detection(
                            output,
                            inputs.input_ids,
                            threshold=args.threshold,
                            text_threshold=0.25,
                            target_sizes=[(image.height, image.width)],
                        )[0]
                    else:
                        result = processor.post_process_grounded_object_detection(
                            output,
                            threshold=args.threshold,
                            target_sizes=[(image.height, image.width)],
                        )[0]
                    boxes, scores = [], []
                    for coords, score in zip(
                        result["boxes"].cpu().tolist(), result["scores"].cpu().tolist()
                    ):
                        try:
                            boxes.append(
                                box_from_xyxy(coords, image.width, image.height)
                            )
                            scores.append(score)
                        except ValueError:
                            continue
                    keep = nms(boxes, scores)
                    boxes, scores = [boxes[i] for i in keep], [scores[i] for i in keep]
            sync(device)
            result = dict(
                base,
                boxes=boxes,
                scores=scores,
                status="truncated" if response_truncated else "ok",
                elapsed_seconds=time.perf_counter() - started,
                image_width=image.width,
                image_height=image.height,
                image_sha256=sha256,
                warmup=(index == 0),
                raw_text=raw_text,
                response_truncated=response_truncated,
                generated_tokens=generated_tokens,
                **({"ended_with_eos": ended_with_eos} if args.model == "qwen" else {}),
            )
            with args.output.open("a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()
            print(
                f"{index+1}/{len(pending)} photo={row['photo_id']} boxes={len(boxes)} seconds={result['elapsed_seconds']:.2f}",
                flush=True,
            )
            consecutive_errors = 0
        except Exception as exc:
            failure = dict(base, status="error", error=f"{type(exc).__name__}: {exc}")
            failure["raw_text"] = raw_text
            if (
                image is not None
                and sha256 is not None
                and inference_started is not None
            ):
                failure.update(
                    boxes=[],
                    scores=[],
                    image_width=image.width,
                    image_height=image.height,
                    image_sha256=sha256,
                    elapsed_seconds=time.perf_counter() - inference_started,
                    warmup=(index == 0),
                    generated_tokens=generated_tokens,
                    ended_with_eos=ended_with_eos,
                    response_truncated=bool(generated_tokens and not ended_with_eos),
                )
            with args.output.with_suffix(".errors.jsonl").open("a") as f:
                f.write(json.dumps(failure) + "\n")
            print(f"ERROR photo={row['photo_id']}: {exc}", flush=True)
            consecutive_errors += 1
            if consecutive_errors >= args.max_consecutive_errors:
                print(
                    "Stopping after consecutive errors; successful predictions are saved",
                    flush=True,
                )
                break
    completed = (
        {r["photo_id"] for r in read_jsonl(args.output)}
        if args.output.exists()
        else set()
    )
    missing = {r["photo_id"] for r in rows} - completed
    if missing:
        raise SystemExit(f"{len(missing)} images missing; rerun same command to retry")


def _size(image, maximum):
    scale = min(1.0, maximum / max(image.size))
    return max(1, round(image.width * scale)), max(1, round(image.height * scale))


if __name__ == "__main__":
    main()
