from pathlib import Path
import re

source = Path("src/pipeline_parallel_compare.py")
s = source.read_text()

# Add --encoder-repeats before argument parsing.
m = re.search(
    r'(?m)^a\s*=\s*(\w+)\.parse_args\(\)\s*$',
    s
)
assert m, "Could not locate argument parsing"

parser = m.group(1)
insertion = (
    f'{parser}.add_argument('
    '"--encoder-repeats", '
    'type=int, default=1)\n'
)

s = s[:m.start()] + insertion + s[m.start():]

# Replace only the vision inference block.
pattern = (
    r"(?m)^    out = vr\[device\]\.infer"
    r"\(\{'pixel_values': frame\}\)\n"
    r"    visual = np\.array"
    r"\(next\(iter\(out\.values\(\)\)\),\s*copy=True\)"
)

replacement = """    out = vr[device].infer({'pixel_values': frame})
    visual = np.array(next(iter(out.values())), copy=True)

    # Synthetic additional encoder work.
    # The decoder receives only the first visual embedding.
    for _ in range(a.encoder_repeats - 1):
        vr[device].infer({'pixel_values': frame})"""

s, count = re.subn(pattern, replacement, s)

assert count == 1, (
    f"Expected one vision inference block, found {count}"
)

out = Path("src/pipeline_vision_work_sweep.py")
out.write_text(s)

print("Created:", out)
print("Original pipeline unchanged.")
