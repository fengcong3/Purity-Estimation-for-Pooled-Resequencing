#!/usr/bin/env python3
"""SNP Depth Analysis Pipeline

This script analyzes the depth support for genotypes at SNP positions:
1. Reads BAM and VCF files
2. Randomly samples SNP positions from VCF during streaming (reservoir sampling)
3. Calculates genotype depth support from BAM file
4. Outputs TSV file with position, genotype, depths, and ratios
5. Generates distribution plots for depth ratios by genotype
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import random
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pysam
import seaborn as sns

LOGGER = logging.getLogger("snp_depth_analysis")


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def genotype_to_code(gt: Tuple[int, int]) -> int:
    """Convert genotype tuple to integer code.
    
    Returns:
        0: REF homozygous (0/0)
        1: ALT homozygous (1/1)
        2: Heterozygous (0/1 or 1/0)
        -1: Invalid/missing
    """
    if gt[0] == 0 and gt[1] == 0:
        return 0
    elif gt[0] == 1 and gt[1] == 1:
        return 1
    elif (gt[0] == 0 and gt[1] == 1) or (gt[0] == 1 and gt[1] == 0):
        return 2
    else:
        return -1


def reservoir_sample_snps(
    vcf_path: pathlib.Path,
    standard_sample: str,
    num_snps: int,
    seed: int = 42,
) -> List[Tuple[str, int, str, str, int]]:
    """Randomly sample SNPs from VCF using reservoir sampling.
    
    This method allows sampling without loading all SNPs into memory first.
    
    Args:
        vcf_path: Path to VCF file
        standard_sample: Sample ID to extract genotype from
        num_snps: Number of SNPs to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of tuples: (chrom, pos, ref, alt, genotype_code)
    """
    random.seed(seed)
    vcf = pysam.VariantFile(str(vcf_path))
    
    if standard_sample not in vcf.header.samples:
        raise KeyError(f"Standard sample {standard_sample} not found in VCF")
    
    LOGGER.info("Scanning VCF and performing reservoir sampling...")
    
    reservoir: List[Tuple[str, int, str, str, int]] = []
    count = 0
    
    for rec in vcf.fetch():
        # Only process biallelic SNPs
        if len(rec.alts) != 1:
            continue
            
        sample_data = rec.samples[standard_sample]
        gt = sample_data.get("GT")
        if gt is None or len(gt) != 2:
            continue
            
        gt_code = genotype_to_code(gt)
        if gt_code == -1:  # Invalid genotype
            continue
        
        chrom = rec.chrom
        pos = rec.pos
        ref = rec.ref.upper()
        alt = rec.alts[0].upper()
        
        count += 1
        
        # Reservoir sampling algorithm
        if len(reservoir) < num_snps:
            reservoir.append((chrom, pos, ref, alt, gt_code))
        else:
            # Randomly replace elements with decreasing probability
            j = random.randint(0, count - 1)
            if j < num_snps:
                reservoir[j] = (chrom, pos, ref, alt, gt_code)
        
        if count % 100000 == 0:
            LOGGER.info("Processed %d SNPs, reservoir size: %d", count, len(reservoir))
    
    vcf.close()
    
    LOGGER.info("Total SNPs scanned: %d", count)
    LOGGER.info("Sampled SNPs: %d", len(reservoir))
    
    return reservoir


def calculate_depth_support(
    bam_path: pathlib.Path,
    snps: List[Tuple[str, int, str, str, int]],
) -> List[Dict[str, any]]:
    """Calculate depth support for each SNP position.
    
    Args:
        bam_path: Path to BAM file
        snps: List of SNP information
        
    Returns:
        List of dicts with position info and depth statistics
    """
    bam = pysam.AlignmentFile(str(bam_path), "rb")
    
    results = []
    
    LOGGER.info("Calculating depth support for %d SNPs...", len(snps))
    
    for idx, (chrom, pos, ref, alt, gt_code) in enumerate(snps):
        ref_count = 0
        alt_count = 0
        other_count = 0
        
        # Pileup at this position
        for pileup_col in bam.pileup(
            chrom,
            pos - 1,
            pos,
            truncate=True,
            stepper="nofilter",
            max_depth=1_000_000,
        ):
            if pileup_col.pos == pos - 1:
                for pileup_read in pileup_col.pileups:
                    if pileup_read.is_del or pileup_read.is_refskip:
                        continue
                    seq = pileup_read.alignment.query_sequence
                    if seq is None or pileup_read.query_position is None:
                        continue
                    base = seq[pileup_read.query_position].upper()
                    
                    if base == ref:
                        ref_count += 1
                    elif base == alt:
                        alt_count += 1
                    else:
                        other_count += 1
                break
        
        total_depth = ref_count + alt_count + other_count
        
        # Calculate genotype support depth based on standard sample genotype
        if gt_code == 0:  # REF homozygous (0/0)
            genotype_support_depth = ref_count
            genotype_str = "0/0"
        elif gt_code == 1:  # ALT homozygous (1/1)
            genotype_support_depth = alt_count
            genotype_str = "1/1"
        elif gt_code == 2:  # Heterozygous (0/1)
            genotype_support_depth = ref_count + alt_count
            genotype_str = "0/1"
        else:
            genotype_support_depth = 0
            genotype_str = "."
        
        # Calculate ratio
        depth_ratio = genotype_support_depth / total_depth if total_depth > 0 else 0.0
        
        result = {
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "genotype": genotype_str,
            "genotype_code": gt_code,
            "total_depth": total_depth,
            "ref_depth": ref_count,
            "alt_depth": alt_count,
            "genotype_support_depth": genotype_support_depth,
            "depth_ratio": depth_ratio,
        }
        
        results.append(result)
        
        if (idx + 1) % 1000 == 0:
            LOGGER.info("Processed %d/%d SNPs", idx + 1, len(snps))
    
    bam.close()
    
    LOGGER.info("Completed depth calculation for all SNPs")
    
    return results


def save_results(
    results: List[Dict[str, any]],
    output_path: pathlib.Path,
) -> None:
    """Save results to TSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output TSV file
    """
    with output_path.open("w", encoding="utf-8") as f:
        # Write header
        f.write("chrom\tpos\tref\talt\tgenotype\ttotal_depth\tref_depth\talt_depth\t"
                "genotype_support_depth\tdepth_ratio\n")
        
        # Write data
        for r in results:
            f.write(
                f"{r['chrom']}\t{r['pos']}\t{r['ref']}\t{r['alt']}\t{r['genotype']}\t"
                f"{r['total_depth']}\t{r['ref_depth']}\t{r['alt_depth']}\t"
                f"{r['genotype_support_depth']}\t{r['depth_ratio']:.6f}\n"
            )
    
    LOGGER.info("Results saved to %s", output_path)


def plot_depth_ratio_distributions(
    results: List[Dict[str, any]],
    output_dir: pathlib.Path,
) -> None:
    """Generate distribution plots for depth ratios.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    # Prepare data
    genotype_labels = {0: "0/0 (REF)", 1: "1/1 (ALT)", 2: "0/1 (HET)"}
    data_by_genotype = {0: [], 1: [], 2: []}
    all_ratios = []
    
    for r in results:
        gt_code = r["genotype_code"]
        ratio = r["depth_ratio"]
        if gt_code in data_by_genotype:
            data_by_genotype[gt_code].append(ratio)
        all_ratios.append(ratio)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot 1: Distribution by genotype (separate subplots)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Depth Ratio Distribution by Genotype", fontsize=16, fontweight="bold")
    
    for idx, (gt_code, label) in enumerate(genotype_labels.items()):
        ax = axes[idx]
        data = data_by_genotype[gt_code]
        
        if len(data) > 0:
            ax.hist(data, bins=50, edgecolor="black", alpha=0.7, color=f"C{idx}")
            ax.axvline(np.mean(data), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(data):.3f}")
            ax.axvline(np.median(data), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(data):.3f}")
            ax.set_xlabel("Depth Ratio", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(f"{label} (n={len(data)})", fontsize=14)
            ax.legend()
            ax.set_xlim(0, 1)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{label} (n=0)", fontsize=14)
    
    plt.tight_layout()
    plot1_path = output_dir / "depth_ratio_by_genotype.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved plot: %s", plot1_path)
    
    # Plot 2: Overall distribution (all genotypes combined)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_ratios, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(np.mean(all_ratios), color="red", linestyle="--", linewidth=2, 
               label=f"Mean: {np.mean(all_ratios):.3f}")
    ax.axvline(np.median(all_ratios), color="green", linestyle="--", linewidth=2,
               label=f"Median: {np.median(all_ratios):.3f}")
    ax.set_xlabel("Depth Ratio", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(f"Overall Depth Ratio Distribution (n={len(all_ratios)})", 
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plot2_path = output_dir / "depth_ratio_overall.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved plot: %s", plot2_path)
    
    # Plot 3: Violin plot comparing genotypes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for violin plot
    violin_data = []
    violin_labels = []
    for gt_code, label in genotype_labels.items():
        data = data_by_genotype[gt_code]
        if len(data) > 0:
            violin_data.append(data)
            violin_labels.append(f"{label}\n(n={len(data)})")
    
    if len(violin_data) > 0:
        parts = ax.violinplot(violin_data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(violin_labels) + 1))
        ax.set_xticklabels(violin_labels, fontsize=12)
        ax.set_ylabel("Depth Ratio", fontsize=14)
        ax.set_title("Depth Ratio Distribution Comparison", fontsize=16, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plot3_path = output_dir / "depth_ratio_violin.png"
    plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved plot: %s", plot3_path)
    
    # Plot 4: Box plot comparing genotypes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(violin_data) > 0:
        bp = ax.boxplot(violin_data, labels=violin_labels, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        # Color the boxes
        colors = ["C0", "C1", "C2"]
        for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel("Depth Ratio", fontsize=14)
        ax.set_title("Depth Ratio Distribution (Box Plot)", fontsize=16, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plot4_path = output_dir / "depth_ratio_boxplot.png"
    plt.savefig(plot4_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved plot: %s", plot4_path)
    
    # Print summary statistics
    LOGGER.info("\n=== Summary Statistics ===")
    LOGGER.info("Overall (all genotypes):")
    LOGGER.info("  Mean: %.4f", np.mean(all_ratios))
    LOGGER.info("  Median: %.4f", np.median(all_ratios))
    LOGGER.info("  Std: %.4f", np.std(all_ratios))
    
    for gt_code, label in genotype_labels.items():
        data = data_by_genotype[gt_code]
        if len(data) > 0:
            LOGGER.info("\n%s:", label)
            LOGGER.info("  Count: %d", len(data))
            LOGGER.info("  Mean: %.4f", np.mean(data))
            LOGGER.info("  Median: %.4f", np.median(data))
            LOGGER.info("  Std: %.4f", np.std(data))
            LOGGER.info("  Min: %.4f", np.min(data))
            LOGGER.info("  Max: %.4f", np.max(data))


def main(argv: Optional[List[str]] = None) -> int:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze SNP depth support from BAM and VCF files"
    )
    parser.add_argument(
        "--bam",
        required=True,
        type=pathlib.Path,
        help="Path to BAM file (can be pooled BAM from pipeline)"
    )
    parser.add_argument(
        "--vcf",
        required=True,
        type=pathlib.Path,
        help="Path to VCF file"
    )
    parser.add_argument(
        "--standard-sample",
        required=True,
        help="Sample ID in VCF to use as standard sample"
    )
    parser.add_argument(
        "--num-snps",
        type=int,
        default=10000,
        help="Number of SNPs to randomly sample (default: 10000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("snp_depth_analysis"),
        help="Output directory (default: snp_depth_analysis)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args(argv)
    
    setup_logging(args.log_level)
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Check input files exist
    if not args.bam.exists():
        LOGGER.error("BAM file not found: %s", args.bam)
        return 1
    if not args.vcf.exists():
        LOGGER.error("VCF file not found: %s", args.vcf)
        return 1
    
    LOGGER.info("Starting SNP depth analysis...")
    LOGGER.info("BAM file: %s", args.bam)
    LOGGER.info("VCF file: %s", args.vcf)
    LOGGER.info("Standard sample: %s", args.standard_sample)
    LOGGER.info("Number of SNPs to sample: %d", args.num_snps)
    LOGGER.info("Output directory: %s", args.outdir)
    
    # Step 1: Reservoir sample SNPs from VCF
    snps = reservoir_sample_snps(
        args.vcf,
        args.standard_sample,
        args.num_snps,
        args.seed
    )
    
    if len(snps) == 0:
        LOGGER.error("No valid SNPs found in VCF")
        return 1
    
    # Step 2: Calculate depth support from BAM
    results = calculate_depth_support(args.bam, snps)
    
    # Step 3: Save results to TSV
    output_tsv = args.outdir / "snp_depth_results.tsv"
    save_results(results, output_tsv)
    
    # Step 4: Generate plots
    plot_depth_ratio_distributions(results, args.outdir)
    
    LOGGER.info("\nAnalysis complete!")
    LOGGER.info("Results saved to: %s", args.outdir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
