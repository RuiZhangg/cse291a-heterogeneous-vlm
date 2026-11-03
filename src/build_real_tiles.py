from pathlib import Path

src = Path('src/pipeline_parallel_compare.py')
s = src.read_text()

anchor = 'a = ap.parse_args()'
assert s.count(anchor) == 1
s = s.replace(anchor, '''ap.add_argument('--tiles', type=int, choices=[1, 2, 4], default=1)
ap.add_argument('--thumbnail', action='store_true',
                help='Add full-image thumbnail after grid patches')
''' + anchor, 1)

start = s.index('def preprocess(path):')
end = s.index('\nframes = [preprocess(p) for p in frame_paths]', start)
s = s[:start] + '''def preprocess(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size

    if a.tiles == 1:
        crops = [img]
        grid = (1, 1)
    else:
        # Exact number of spatial tiles, keeping grid aspect ratio
        # as close to the original image as possible.
        candidates = [(cols, a.tiles // cols)
                      for cols in range(1, a.tiles + 1)
                      if a.tiles % cols == 0]
        cols, rows = min(candidates,
                         key=lambda z: abs((z[0] / z[1]) - (w / h)))
        grid = (cols, rows)
        large = img.resize((cols * 448, rows * 448),
                           Image.Resampling.BICUBIC)
        crops = [large.crop((col * 448, row * 448,
                             (col + 1) * 448, (row + 1) * 448))
                 for row in range(rows) for col in range(cols)]
        if a.thumbnail:
            crops.append(img)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    output = []
    for crop in crops:
        x = np.asarray(crop.resize((448, 448), Image.Resampling.BICUBIC),
                       dtype=np.float32) / 255.0
        x = ((x - mean) / std).transpose(2, 0, 1)
        output.append(np.ascontiguousarray(x[None], dtype=np.float32))
    return output
''' + s[end:]

start = s.index('prompt = raw.replace(')
end = s.index('\nids = np.asarray(', start)
s = s[:start] + '''tile_count = len(frames[0])
assert all(len(frame) == tile_count for frame in frames)
print('Image tiles per frame:', tile_count, 'grid tiles:', a.tiles,
      'thumbnail:', bool(a.thumbnail), flush=True)
prompt = raw.replace(
    '<image>', '<img>' + '<IMG_CONTEXT>' * (256 * tile_count) + '</img>'
)
''' + s[end:]

assert s.count('assert int(positions.sum()) == 256') == 1
s = s.replace('assert int(positions.sum()) == 256',
              'assert int(positions.sum()) == 256 * tile_count', 1)

start = s.index('    out = vr[device].infer(', s.index('def encode(frame, device):'))
end = s.index('\n    return visual,', start)
s = s[:start] + '''    visual_tiles = []
    for tile in frame:
        out = vr[device].infer({'pixel_values': tile})
        features = np.array(next(iter(out.values())), copy=True)
        assert features.shape == (1, 256, 896), features.shape
        visual_tiles.append(features)
    visual = np.concatenate(visual_tiles, axis=0)
    assert visual.shape == (tile_count, 256, 896)
''' + s[end:]

out = Path('src/pipeline_real_tiles.py')
out.write_text(s)
print('Created', out)
print('Source unchanged', src)
print('Tile counts: 1,2,4; optional thumbnail adds 1 to counts >1')
