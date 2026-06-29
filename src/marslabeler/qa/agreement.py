"""Inter-labeler agreement analysis (Cohen's kappa, confusion matrix)."""

from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq


def calculate_cohens_kappa(confusion_matrix: np.ndarray) -> float:
    """
    Calculate Cohen's kappa from a confusion matrix.

    Args:
        confusion_matrix: Square confusion matrix (rows: labeler1, cols: labeler2)

    Returns:
        Kappa score (-1 to 1, where 1 is perfect agreement)
    """
    n = confusion_matrix.sum()
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = np.trace(confusion_matrix) / n

    # Expected agreement (marginal probabilities)
    p_e = np.sum((confusion_matrix.sum(axis=0) / n) * (confusion_matrix.sum(axis=1) / n))

    # Cohen's kappa
    if p_e == 1.0:
        return 0.0  # No disagreement to measure
    return (p_o - p_e) / (1 - p_e)


def compare_labelers(
    parquet_path1: Path,
    parquet_path2: Path,
    obs_id: str,
) -> dict:
    """
    Compare labels from two labelers for the same observation.

    Args:
        parquet_path1: Path to first labeler's Parquet file
        parquet_path2: Path to second labeler's Parquet file
        obs_id: Observation ID to match

    Returns:
        Dict with agreement metrics and confusion matrix
    """
    # Load both tables
    table1 = pq.read_table(str(parquet_path1))
    table2 = pq.read_table(str(parquet_path2))

    # Convert to dicts keyed by block_id
    blocks1 = {}
    for i in range(len(table1)):
        row = table1.slice(i, 1).to_pydict()
        block_id = row["block_id"][0]
        blocks1[block_id] = row["class_id"][0]

    blocks2 = {}
    for i in range(len(table2)):
        row = table2.slice(i, 1).to_pydict()
        block_id = row["block_id"][0]
        blocks2[block_id] = row["class_id"][0]

    # Find common blocks
    common_blocks = set(blocks1.keys()) & set(blocks2.keys())
    if not common_blocks:
        return {"error": "No common blocks"}

    # Build confusion matrix
    labels1 = [blocks1[bid] for bid in sorted(common_blocks)]
    labels2 = [blocks2[bid] for bid in sorted(common_blocks)]

    # Get unique class IDs
    unique_classes = sorted(set(labels1) | set(labels2))
    class_to_idx = {cls_id: i for i, cls_id in enumerate(unique_classes)}

    # Build confusion matrix
    cm = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    for label1, label2 in zip(labels1, labels2):
        i = class_to_idx[label1]
        j = class_to_idx[label2]
        cm[i, j] += 1

    # Calculate metrics
    kappa = calculate_cohens_kappa(cm)
    agreement = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0

    # Disagreements
    disagreements = []
    for bid in sorted(common_blocks):
        if blocks1[bid] != blocks2[bid]:
            disagreements.append({
                "block_id": bid,
                "labeler1_class": blocks1[bid],
                "labeler2_class": blocks2[bid],
            })

    return {
        "obs_id": obs_id,
        "total_blocks": len(common_blocks),
        "agreement_pct": agreement * 100,
        "cohens_kappa": kappa,
        "confusion_matrix": cm,
        "class_ids": unique_classes,
        "disagreements": disagreements,
        "num_disagreements": len(disagreements),
    }


def format_agreement_report(result: dict) -> str:
    """Format agreement analysis as readable report."""
    if "error" in result:
        return f"Error: {result['error']}"

    lines = [
        f"Agreement Analysis: {result['obs_id']}",
        f"Blocks compared: {result['total_blocks']}",
        f"Overall agreement: {result['agreement_pct']:.1f}%",
        f"Cohen's kappa: {result['cohens_kappa']:.3f}",
        f"Disagreements: {result['num_disagreements']}",
    ]

    if result["num_disagreements"] > 0 and result["num_disagreements"] <= 20:
        lines.append("")
        lines.append("Disagreement blocks:")
        for d in result["disagreements"][:20]:
            lines.append(
                f"  {d['block_id']}: "
                f"L1={d['labeler1_class']} vs L2={d['labeler2_class']}"
            )

    return "\n".join(lines)
