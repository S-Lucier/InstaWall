import json
from pathlib import Path

# Check multiple JSON files for any circular room indicators
json_dir = Path(r'C:\Users\shini\Downloads\watabou_exports')

for json_file in list(json_dir.glob('*.json'))[:3]:  # Check first 3
    print(f"\n{'='*60}")
    print(f"File: {json_file.name}")
    print('='*60)

    with open(json_file) as f:
        data = json.load(f)

    print(f"Top-level keys: {list(data.keys())}")

    # Check all unique keys in rects
    all_rect_keys = set()
    for r in data['rects']:
        all_rect_keys.update(r.keys())

    print(f"\nAll rect properties: {all_rect_keys}")

    # Show a few example rects with all their properties
    print("\nSample rects:")
    for i, rect in enumerate(data['rects'][:5]):
        print(f"  {i}: {rect}")

    # Check if there are any rects with extra properties
    rects_with_extra = [r for r in data['rects'] if len(r.keys()) > 4]
    if rects_with_extra:
        print(f"\nRects with extra properties: {len(rects_with_extra)}")
        print("Examples:")
        for r in rects_with_extra[:3]:
            print(f"  {r}")
