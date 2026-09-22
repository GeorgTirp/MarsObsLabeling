"""Exercise real Qt labeling, persistence, export, and a tiny CPU training batch.

Run with a GUI+torch environment and QT_QPA_PLATFORM=offscreen if desired.
All trial labels go into --output, never the configured production labels folder.
The assigned classes are mechanical QA examples, NOT scientific annotations.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import rasterio
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from marslabeler.ui.mainwindow import MainWindow
from marslabeler.model.export import export_coarse_geotiff
from export_labels import export_probe_set


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo / 'configs/app.yaml').read_text())
    config['paths']['classes_file'] = str(repo / 'configs/classes.yaml')
    config['paths']['labels_dir'] = str(output / 'labels')
    config['autosave'] = {'every_n_labels': 3, 'every_seconds': 30}
    config_path = output / 'app.yaml'
    config_path.write_text(yaml.safe_dump(config))
    app = QApplication.instance() or QApplication([])
    errors = []
    # Do not leave an unattended verification stuck in an error modal.
    original_critical = QMessageBox.critical
    QMessageBox.critical = lambda *args: errors.append(args[2])

    def opened():
        win = MainWindow(config_path, match_training_gsd=False)
        win.help_shown_on_startup = True
        win.show()
        win._load_observation(source)
        app.processEvents()
        assert not errors, errors
        assert win.session is not None
        return win

    print('Opening actual observation through the GUI loader...', flush=True)
    started = time.monotonic()
    win = opened()
    grid = win.session.grid
    candidates = [b for b in grid.iter_blocks() if b.w_px > 0 and b.h_px > 0
                  and win.skip_decisions[b.block_id]['nodata_fraction'] < 0.1]
    assert len(candidates) >= 6
    # Spread the trial over the observation, independent of panel-major ordering.
    targets = [candidates[i] for i in np.linspace(0, len(candidates)-1, 6, dtype=int)]
    expected = {}
    class_ids = [0, 11, 12, 17, 6, 13]
    for index, (block, cid) in enumerate(zip(targets, class_ids)):
        win._enter_panel(block.panel_idx)
        app.processEvents()
        # Click the actual rendered tile, not just its label-store index.
        side = win.canvas.canvas_width / grid.blocks_per_panel_col
        point = win.canvas.mapFromScene(QPointF((block.block_col + 0.5) * side,
                                                (block.block_row + 0.5) * side))
        QTest.mouseClick(win.canvas.viewport(), Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, point)
        app.processEvents()
        assert win.session.current_block().block_id == block.block_id
        key = win.classes_scheme.classes[cid].hotkey
        QTest.keyClicks(win, key)
        app.processEvents()
        assert win.session.labels.get_record(block.block_id).class_id == cid
        expected[block.block_id] = cid
        if index == 2:
            # No explicit save yet: this must be the count-based autosave.
            saved = pq.read_table(output / 'labels' / f'{grid.obs_id}.parquet').to_pylist()
            assert sum(r['status'] == 'labeled' for r in saved) == 3
        print(f'Click + hotkey: {block.block_id} -> {cid}', flush=True)
    # Clear/undo/redo and relabel after the count-based save, then immediately close.
    first = targets[0]
    win.session.move_to_block(grid.block_index_at_pixel(first.x_px, first.y_px))
    win._refresh_view()
    QTest.keyClick(win, Qt.Key_Backspace)
    QTest.keyClick(win, Qt.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert win.session.labels.get_record(first.block_id).class_id == 0
    QTest.keyClick(win, Qt.Key_Z, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert win.session.labels.get_record(first.block_id).class_id == -3
    QTest.keyClicks(win, win.classes_scheme.classes[1].hotkey)
    expected[first.block_id] = 1
    win.grab().save(str(output / 'gui_labeled.png'))
    cursor = win.session.current_block_idx
    assert win.close()
    assert not errors, errors
    print('Reopening saved session...', flush=True)
    resumed = opened()
    assert resumed.session.current_block_idx == cursor
    assert resumed.session.labels.count_labeled() == 6
    for bid, cid in expected.items():
        assert resumed.session.labels.get_record(bid).class_id == cid
    resumed.grab().save(str(output / 'gui_reopened.png'))
    parquet = output / 'labels' / f'{grid.obs_id}.parquet'
    export_probe_set(source, parquet, Path(config['paths']['classes_file']), output / 'probe')
    export_coarse_geotiff(resumed.session.labels, resumed.session.grid, output / 'coarse.tif')
    with (output / 'probe/labels.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(expected) == len(list((output / 'probe/crops').glob('*.png')))
    checks = []
    with rasterio.open(source) as raster, rasterio.open(output / 'coarse.tif') as coarse:
        coarse_data = coarse.read(1)
        for row in rows:
            assert None not in row, row
            bid = row['block_id']
            x, y, w, h = (int(row[k]) for k in ('x_px', 'y_px', 'w_px', 'h_px'))
            cid = int(row['class_id'])
            native = raster.read(1, window=rasterio.windows.Window(x, y, w, h))
            with Image.open(output / 'probe/crops' / row['crop_filename']) as image:
                exported = np.asarray(image)
            np.testing.assert_array_equal(exported, native)
            assert exported.dtype == native.dtype
            assert cid == expected[bid]
            assert row['class_name'] == resumed.classes_scheme.get_name(cid)
            assert coarse_data[y // grid.block_size, x // grid.block_size] == cid
            assert coarse.transform * (x/grid.block_size, y/grid.block_size) == raster.transform * (x, y)
            checks.append(dict(block_id=bid, x_px=x, y_px=y, w_px=w, h_px=h,
                               class_id=cid, class_name=row['class_name'], dtype=str(native.dtype),
                               pixel_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                               max_absolute_pixel_error=0))
    print('All six crops match the source exactly; CSV and GeoTIFF ids agree.', flush=True)

    class ProbeDataset(Dataset):
        def __len__(self):
            return len(rows)
        def __getitem__(self, index):
            row = rows[index]
            with Image.open(output / 'probe/crops' / row['crop_filename']) as image:
                raw = np.asarray(image).copy()
            tensor = torch.from_numpy(raw.astype(np.float32) / np.iinfo(raw.dtype).max)[None, None]
            tensor = F.interpolate(tensor, size=(64, 64), mode='bilinear', align_corners=False)[0]
            return tensor, int(row['class_id']), row['block_id']

    torch.manual_seed(0)
    loader = DataLoader(ProbeDataset(), batch_size=6, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    images, labels, ids = next(iter(loader))
    assert labels.tolist() == [expected[bid] for bid in ids]
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                torch.nn.Linear(8, len(resumed.classes_scheme.classes)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model[0].weight.detach().clone()
    loss = F.cross_entropy(model(images), labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model[0].weight)
    print(f'Shuffled training batch: {tuple(images.shape)}, finite loss={float(loss.detach()):.6f}, weights updated.', flush=True)

    # A contact sheet for visual review; stretch only this preview, never PNG crops.
    sheet = Image.new('RGB', (3 * 320, 2 * 355), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        with Image.open(output / 'probe/crops' / row['crop_filename']) as image:
            raw = np.asarray(image).astype(float)
        valid = raw[raw > 0]
        lo, hi = np.percentile(valid, (1, 99))
        preview = np.clip((raw-lo) / max(hi-lo, 1) * 255, 0, 255).astype(np.uint8)
        thumb = Image.fromarray(preview).resize((300, 300))
        px, py = (i % 3) * 320 + 10, (i // 3) * 355 + 5
        sheet.paste(thumb, (px, py))
        draw.text((px, py + 303), f"QA class {row['class_id']} | ({row['x_px']}, {row['y_px']})", fill='black')
        draw.text((px, py + 318), row['class_name'], fill='black')
    sheet.save(output / 'verified_crops.png')
    report = {'source': str(source), 'source_fingerprint': resumed.session.raster.fingerprint(),
              'note': 'Mechanical QA labels only, not scientific terrain annotations.',
              'labels_verified': 6, 'workflow': ['GUI click', 'class hotkey', 'count autosave',
              'clear', 'undo', 'redo', 'edit', 'close save', 'reopen', 'CSV/PNG export',
              'exact source pixel comparison', 'GeoTIFF position/class check',
              'shuffled PyTorch batch', 'backward and optimizer step'],
              'crop_checks': checks, 'training_batch_shape': list(images.shape),
              'training_labels': labels.tolist(), 'training_loss': float(loss.detach()),
              'elapsed_seconds': time.monotonic() - started}
    (output / 'verification.json').write_text(json.dumps(report, indent=2))
    assert resumed.close()
    QMessageBox.critical = original_critical
    print(f'PASS: report and screenshots at {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
