
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw
from preprocessing.duplicate_detection import dhash_image

ROOT = Path(__file__).resolve().parent

PROCESSED = ROOT / "data" / "processed"
REPORTS = PROCESSED / "reports"
METADATA = PROCESSED / "metadata"

SUMMARY = json.loads(
    (METADATA / "dataset_summary.json").read_text(encoding="utf-8")
)

DUPLICATES = json.loads(
    (REPORTS / "duplicate_report.json").read_text(encoding="utf-8")
)

CLASS_ROOT = Path(SUMMARY["class_root"])
OUTPUT = REPORTS / "duplicate_diagnostics"
OUTPUT.mkdir(parents=True, exist_ok=True)


def hamming_distance(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count()


def image_hash(path):
    with Image.open(path) as img:
        return dhash_image(img)


def make_contact_sheet(group, distances, output_path):
    members = group["members"]

    # Select up to 12 evenly distributed members.
    count = min(12, len(members))
    indices = sorted({
        round(i * (len(members) - 1) / max(count - 1, 1))
        for i in range(count)
    })

    selected = [members[i] for i in indices]

    thumb_w, thumb_h = 150, 175
    columns = 4
    rows = (len(selected) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (columns * thumb_w, rows * thumb_h),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for position, rel_path in enumerate(selected):
        path = CLASS_ROOT / rel_path

        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((130, 130))

        col = position % columns
        row = position // columns
        x = col * thumb_w
        y = row * thumb_h

        sheet.paste(img, (x + 10, y + 5))

        distance = distances[rel_path]
        label = f"{Path(rel_path).name}"
        draw.text((x + 5, y + 140), label, fill="black")
        draw.text(
            (x + 5, y + 155),
            f"Anchor distance: {distance}",
            fill="black"
        )

    sheet.save(output_path)


groups_by_id = {
    group["group_id"]: group
    for group in DUPLICATES["groups"]
}

print("LARGEST GROUP DIAGNOSTICS")
print("=" * 75)

for item in DUPLICATES["largest_groups"][:5]:
    group = groups_by_id[item["group_id"]]
    members = group["members"]

    # First member is used as the reference anchor.
    anchor_path = CLASS_ROOT / members[0]
    anchor_hash = image_hash(anchor_path)

    distances = {}
    for rel_path in members:
        current_hash = image_hash(CLASS_ROOT / rel_path)
        distances[rel_path] = hamming_distance(
            anchor_hash, current_hash
        )

    values = list(distances.values())

    print(f"\nGroup: {group['group_id']}")
    print(f"Class: {group['class_name']}")
    print(f"Size: {len(members)}")
    print(f"Anchor: {members[0]}")
    print(f"Anchor distance min: {min(values)}")
    print(f"Anchor distance median: {statistics.median(values)}")
    print(f"Anchor distance max: {max(values)}")
    print(
        "Members within threshold 4:",
        sum(value <= 4 for value in values),
        "/",
        len(values)
    )

    sheet_path = OUTPUT / f"{group['group_id']}_contact_sheet.jpg"
    make_contact_sheet(group, distances, sheet_path)

    print(f"Contact sheet: {sheet_path}")

print("\nDiagnostics completed.")
