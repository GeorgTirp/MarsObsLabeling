"""Label export: convert label store to GeoTIFF."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from marslabeler.model.grid import Grid
from marslabeler.model.labelstore import LabelStore


def export_coarse_geotiff(
    label_store: LabelStore,
    grid: Grid,
    output_path: Path,
    nodata_value: int = 255,
) -> None:
    """
    Export labels as a coarse GeoTIFF (one pixel per block).

    Args:
        label_store: The label store with assigned classes
        grid: The grid geometry
        output_path: Output GeoTIFF path
        nodata_value: Value to use for unlabeled/nodata blocks (typically 255 for uint8)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute output grid dimensions
    blocks_across = grid.panels_across * grid.blocks_per_panel_col
    blocks_down = grid.panels_down * grid.blocks_per_panel_row

    # Build output array (one pixel per block)
    data = np.full((blocks_down, blocks_across), nodata_value, dtype=np.uint8)

    for block_idx, block in enumerate(grid.iter_blocks()):
        record = label_store.get_record(block.block_id)

        # Calculate output pixel coords
        panel_row, panel_col = divmod(block.panel_idx, grid.panels_across)
        output_row = panel_row * grid.blocks_per_panel_row + block.block_row
        output_col = panel_col * grid.blocks_per_panel_col + block.block_col

        # Assign class_id (offset nodata/abstain to valid uint8 range)
        if record.status == "labeled":
            # User-assigned class (>= 0)
            data[output_row, output_col] = record.class_id & 0xFF
        elif record.status == "abstain":
            # Abstain (-1) → map to 253
            data[output_row, output_col] = 253
        elif record.status == "nodata":
            # Nodata (-2) → map to 254
            data[output_row, output_col] = 254
        # else: unlabeled → stays as nodata_value (255)

    # Scale the geotransform: output pixel = block_size * input pixel
    # If source affine is (c, a, 0, f, 0, e), then coarse affine is
    # (c, a*block_size, 0, f, 0, e*block_size)
    src_transform = grid.transform
    scale = grid.block_size
    coarse_transform = Affine(
        src_transform.a * scale,
        src_transform.b,
        src_transform.c,
        src_transform.d,
        src_transform.e * scale,
        src_transform.f,
    )

    # Write GeoTIFF
    kwargs = {
        "driver": "GTiff",
        "height": blocks_down,
        "width": blocks_across,
        "count": 1,
        "dtype": np.uint8,
        "transform": coarse_transform,
        "nodata": nodata_value,
    }
    if grid.crs is not None:
        kwargs["crs"] = grid.crs

    with rasterio.open(str(output_path), "w", **kwargs) as dst:
        dst.write(data, 1)


def export_class_metadata(
    label_store: LabelStore,
    classes_scheme,
    output_path: Path,
) -> None:
    """
    Export class information as JSON (for probe set / training data).

    Args:
        label_store: The label store
        classes_scheme: ClassScheme object
        output_path: Output JSON path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes_list = []
    for class_id, class_obj in sorted(classes_scheme.classes.items()):
        classes_list.append({
            "id": class_id,
            "name": class_obj.name,
            "color": class_obj.color,
            "hotkey": class_obj.hotkey,
        })

    # Add special classes
    classes_list.append({
        "id": classes_scheme.abstain.id,
        "name": classes_scheme.abstain.name,
        "color": classes_scheme.abstain.color,
    })
    classes_list.append({
        "id": classes_scheme.nodata.id,
        "name": classes_scheme.nodata.name,
        "color": classes_scheme.nodata.color,
    })

    with open(output_path, "w") as f:
        import json
        json.dump({"classes": classes_list}, f, indent=2)
