#!/usr/bin/env python3
"""Pipeline for pooled resequencing purity estimation on real data.

This script automates the following steps:

1. Down-sample each CRAM according to a target depth allocation that is
   proportional to the desired mixture weights.
2. Merge the down-sampled BAMs into a single pooled BAM and index it.
3. Filter a VCF to the target chromosome and extract the standard sample genotype.
4. Collect allele counts for the selected SNPs from the pooled BAM.
5. Run the covariance and EM-based purity estimators from ``multi_sample_v8a``.

External tools (samtools / bcftools) can be provided through a JSON config file.
See ``tool_paths.example.json`` for the expected format.

The pipeline records its progress in ``pipeline_state.json`` inside the output
directory. Steps whose outputs already exist and match the recorded metadata are
skipped on subsequent runs unless explicitly re-run with ``--force-step``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pysam

import multi_sample_v8a as msim

LOGGER = logging.getLogger("purity_pipeline")


class PipelineState:
    """Tracks completion of pipeline steps and associated outputs."""

    GLOBAL_KEY = "__global__"

    def __init__(self, path: pathlib.Path, forced: Optional[Dict[str, Set[str]]] = None) -> None:
        self.path = path
        self.forced: Dict[str, Set[str]] = forced or {}
        self.state: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    raw: Dict[str, Dict[str, Dict[str, Any]]] = json.load(handle)
                self.state = raw
            except json.JSONDecodeError:
                LOGGER.warning("State file %s is not valid JSON; starting with an empty state", path)

    def _norm_key(self, key: Optional[str]) -> str:
        return key if key is not None else self.GLOBAL_KEY

    def _is_forced(self, step: str, key: Optional[str]) -> bool:
        forced_keys = self.forced.get(step, set())
        normalized = self._norm_key(key)
        return PipelineState.GLOBAL_KEY in forced_keys or normalized in forced_keys

    def is_forced(self, step: str, key: Optional[str] = None) -> bool:
        return self._is_forced(step, key)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2)

    def is_done(
        self,
        step: str,
        key: Optional[str] = None,
        *,
        expected_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self._is_forced(step, key):
            return False
        step_state = self.state.get(step)
        if not step_state:
            return False
        entry = step_state.get(self._norm_key(key))
        if not entry:
            return False
        outputs = entry.get("outputs", [])
        for output in outputs:
            if not pathlib.Path(output).exists():
                return False
        if expected_metadata is not None:
            stored = entry.get("metadata", {})
            for meta_key, meta_val in expected_metadata.items():
                if stored.get(meta_key) != meta_val:
                    return False
        return True

    def get_metadata(self, step: str, key: Optional[str] = None) -> Dict[str, Any]:
        step_state = self.state.get(step)
        if not step_state:
            return {}
        entry = step_state.get(self._norm_key(key))
        if not entry:
            return {}
        metadata = entry.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        return {}

    def mark_done(
        self,
        step: str,
        key: Optional[str] = None,
        *,
        outputs: Sequence[pathlib.Path] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        step_state = self.state.setdefault(step, {})
        normalized = self._norm_key(key)
        entry = {
            "outputs": [str(path) for path in outputs],
            "metadata": metadata or {},
        }
        step_state[normalized] = entry
        self.save()


class CommandError(RuntimeError):
    """Raised when an external command fails."""


def parse_force_steps(values: Sequence[str]) -> Dict[str, Set[str]]:
    forced: Dict[str, Set[str]] = {}
    for raw in values:
        if not raw:
            continue
        if ":" in raw:
            step, key = raw.split(":", 1)
            step = step.strip()
            key = key.strip()
            if not step:
                continue
            forced.setdefault(step, set()).add(key or PipelineState.GLOBAL_KEY)
        else:
            forced.setdefault(raw.strip(), set()).add(PipelineState.GLOBAL_KEY)
    return forced


def round_float(value: float, digits: int = 12) -> float:
    return round(value, digits)


@dataclass
class SampleConfig:
    name: str
    cram_path: pathlib.Path
    target_bases: float
    total_bases: float
    fraction: float
    bam_path: pathlib.Path
    stats_path: pathlib.Path


def setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_tool_paths(config_path: Optional[pathlib.Path]) -> Dict[str, str]:
    tools = {"samtools": "samtools", "bcftools": "bcftools"}
    if config_path is None:
        return tools
    with config_path.open("r", encoding="utf-8") as handle:
        user_cfg = json.load(handle)
    for key in tools:
        if key in user_cfg and user_cfg[key]:
            tools[key] = str(user_cfg[key])
    return tools


def run_command(cmd: Sequence[str], log_path: pathlib.Path) -> None:
    LOGGER.debug("Running command: %s", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise CommandError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def parse_percentage_file(path: pathlib.Path) -> Dict[str, float]:
    percents: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Malformed line in percentage file: {line}")
            sample, value = parts
            percents[sample] = float(value)
    total = sum(percents.values())
    if not math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        LOGGER.warning("Percentages sum to %.4f (expected 1.0)", total)
    return percents


def locate_cram(sample: str, cram_dir: pathlib.Path) -> pathlib.Path:
    candidates = list(cram_dir.glob(f"{sample}*.cram"))
    if not candidates:
        raise FileNotFoundError(f"Could not locate CRAM for sample {sample} in {cram_dir}")
    if len(candidates) > 1:
        LOGGER.warning("Multiple CRAMs found for %s, using %s", sample, candidates[0])
    return candidates[0]


def extract_bases_mapped(stats_file: pathlib.Path) -> float:
    pattern = re.compile(r"^bases mapped:\s+(\d+)")
    with stats_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    raise ValueError(f"Could not parse bases mapped from {stats_file}")


def collect_sample_configs(
    percentages: Dict[str, float],
    cram_dir: pathlib.Path,
    genome_size: float,
    target_depth: float,
    reference: pathlib.Path,
    outdir: pathlib.Path,
    tools: Dict[str, str],
    threads: int,
    seed: int,
    state: PipelineState,
) -> List[SampleConfig]:
    configs: List[SampleConfig] = []
    for index, (sample, weight) in enumerate(percentages.items()):
        cram_path = locate_cram(sample, cram_dir)
        stats_path = outdir / "stats" / f"{sample}.stats.txt"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        total_bases: Optional[float] = None
        stats_done = state.is_done("samtools_stats", sample)
        forced_stats = state.is_forced("samtools_stats", sample)
        if stats_done:
            LOGGER.info("Stats for %s already recorded at %s; skipping", sample, stats_path)
            meta = state.get_metadata("samtools_stats", sample)
            if "bases_mapped" in meta:
                try:
                    total_bases = float(meta["bases_mapped"])
                except (TypeError, ValueError):
                    total_bases = None
        elif stats_path.exists() and not forced_stats:
            LOGGER.info("Found existing stats output for %s at %s; reusing", sample, stats_path)
            total_bases = extract_bases_mapped(stats_path)
            state.mark_done(
                "samtools_stats",
                sample,
                outputs=[stats_path],
                metadata={"bases_mapped": total_bases},
            )
        else:
            if forced_stats:
                LOGGER.info("Forcing regeneration of stats for %s", sample)
            elif stats_path.exists():
                LOGGER.info("Metadata mismatch for %s stats; recomputing", sample)
            cmd_stats = [
                tools["samtools"],
                "stats",
                "-@",
                str(threads),
                "--reference",
                str(reference),
                str(cram_path),
            ]
            run_command(cmd_stats, stats_path)
            total_bases = extract_bases_mapped(stats_path)
            state.mark_done(
                "samtools_stats",
                sample,
                outputs=[stats_path],
                metadata={"bases_mapped": total_bases},
            )
        if total_bases is None:
            raise ValueError(f"Could not determine mapped bases for sample {sample}")
        target_bases = genome_size * target_depth * weight
        fraction = min(target_bases / total_bases if total_bases else 0.0, 1.0)
        bam_path = outdir / "downsampled" / f"{sample}.subsampled.bam"
        bam_path.parent.mkdir(parents=True, exist_ok=True)
        configs.append(
            SampleConfig(
                name=sample,
                cram_path=cram_path,
                target_bases=target_bases,
                total_bases=total_bases,
                fraction=fraction,
                bam_path=bam_path,
                stats_path=stats_path,
            )
        )
        LOGGER.info(
            "Sample %s: target=%.2f Gb total=%.2f Gb fraction=%.4f",
            sample,
            target_bases / 1e9,
            total_bases / 1e9,
            fraction,
        )
    return configs


def subsample_cram(
    sample_cfg: SampleConfig,
    tools: Dict[str, str],
    reference: pathlib.Path,
    threads: int,
    seed: int,
    state: PipelineState,
) -> None:
    metadata = {
        "fraction": round_float(sample_cfg.fraction),
        "seed": int(seed),
        "reference": str(reference),
    }
    bam_index = sample_cfg.bam_path.with_suffix(sample_cfg.bam_path.suffix + ".bai")
    prev_metadata = state.get_metadata("downsample", sample_cfg.name)
    if state.is_done("downsample", sample_cfg.name, expected_metadata=metadata):
        LOGGER.info(
            "Downsampled BAM for %s already exists at %s; skipping",
            sample_cfg.name,
            sample_cfg.bam_path,
        )
        return
    forced = state.is_forced("downsample", sample_cfg.name)
    if (
        not forced
        and sample_cfg.bam_path.exists()
        and bam_index.exists()
        and (not prev_metadata or prev_metadata == metadata)
    ):
        LOGGER.info(
            "Downsampled BAM for %s found at %s; reusing existing file",
            sample_cfg.name,
            sample_cfg.bam_path,
        )
        state.mark_done(
            "downsample",
            sample_cfg.name,
            outputs=[sample_cfg.bam_path, bam_index],
            metadata=metadata,
        )
        return
    if forced:
        LOGGER.info("Forcing regeneration of downsampled BAM for %s", sample_cfg.name)
    elif prev_metadata and prev_metadata != metadata:
        LOGGER.info("Metadata change detected for %s; regenerating downsampled BAM", sample_cfg.name)
    if sample_cfg.fraction >= 0.999999:
        LOGGER.info("Fraction for %s >= 1.0, converting to BAM without subsampling", sample_cfg.name)
        cmd_view = [
            tools["samtools"],
            "view",
            "-@",
            str(threads),
            "-b",
            "-T",
            str(reference),
            str(sample_cfg.cram_path),
        ]
    else:
        frac = max(sample_cfg.fraction, 0.0)
        frac_int = max(min(int(round(frac * 1_000_000)), 999_999), 0)
        frac_str = f"{seed % 10_000}.{frac_int:06d}"
        cmd_view = [
            tools["samtools"],
            "view",
            "-@",
            str(threads),
            "-b",
            "-s",
            frac_str,
            "-T",
            str(reference),
            str(sample_cfg.cram_path),
        ]
    sort_tmp = sample_cfg.bam_path.parent / f"{sample_cfg.name}.tmp.bam"
    log_path = sample_cfg.bam_path.parent / f"{sample_cfg.name}.view.log"
    with sort_tmp.open("wb") as bam_out, log_path.open("w", encoding="utf-8") as log_file:
        LOGGER.debug("Running command: %s", " ".join(cmd_view))
        proc = subprocess.run(cmd_view, stdout=bam_out, stderr=log_file)
    if proc.returncode != 0:
        raise CommandError(f"samtools view failed for {sample_cfg.name}")
    cmd_sort = [
        tools["samtools"],
        "sort",
        "-@",
        str(threads),
        "-o",
        str(sample_cfg.bam_path),
        str(sort_tmp),
    ]
    sort_log = sample_cfg.bam_path.parent / f"{sample_cfg.name}.sort.log"
    run_command(cmd_sort, sort_log)
    sort_tmp.unlink(missing_ok=True)
    cmd_index = [tools["samtools"], "index", "-@", str(threads), str(sample_cfg.bam_path)]
    index_log = sample_cfg.bam_path.parent / f"{sample_cfg.name}.index.log"
    run_command(cmd_index, index_log)
    state.mark_done(
        "downsample",
        sample_cfg.name,
        outputs=[sample_cfg.bam_path, bam_index],
        metadata=metadata,
    )


def merge_bams(
    configs: Sequence[SampleConfig],
    tools: Dict[str, str],
    outdir: pathlib.Path,
    threads: int,
    state: PipelineState,
) -> pathlib.Path:
    merged_unsorted = outdir / "pooled.unsorted.bam"
    merged_sorted = outdir / "pooled.sorted.bam"
    merge_log = outdir / "merge.log"
    merged_index = merged_sorted.with_suffix(merged_sorted.suffix + ".bai")
    metadata = {
        "inputs": [str(cfg.bam_path) for cfg in configs],
        "threads": int(threads),
        "samples": [cfg.name for cfg in configs],
        "input_metadata": [state.get_metadata("downsample", cfg.name) for cfg in configs],
    }
    if state.is_done("merge_pooled_bam", expected_metadata=metadata):
        LOGGER.info("Merged pooled BAM already exists at %s; skipping merge", merged_sorted)
        return merged_sorted
    prev_metadata = state.get_metadata("merge_pooled_bam")
    forced = state.is_forced("merge_pooled_bam")
    if (
        not forced
        and merged_sorted.exists()
        and merged_index.exists()
        and (not prev_metadata)
    ):
        LOGGER.info("Merged BAM found at %s; reusing existing file", merged_sorted)
        state.mark_done(
            "merge_pooled_bam",
            outputs=[merged_sorted, merged_index],
            metadata=metadata,
        )
        return merged_sorted
    if forced:
        LOGGER.info("Forcing regeneration of pooled BAM at %s", merged_sorted)
    elif prev_metadata and prev_metadata != metadata:
        LOGGER.info("Metadata change detected for pooled BAM; regenerating %s", merged_sorted)
    cmd_merge = [tools["samtools"], "merge", "-@", str(threads), "-f", str(merged_unsorted)]
    cmd_merge.extend(str(cfg.bam_path) for cfg in configs)
    run_command(cmd_merge, merge_log)
    sort_log = outdir / "pooled.sort.log"
    cmd_sort = [tools["samtools"], "sort", "-@", str(threads), "-o", str(merged_sorted), str(merged_unsorted)]
    run_command(cmd_sort, sort_log)
    merged_unsorted.unlink(missing_ok=True)
    index_log = outdir / "pooled.index.log"
    cmd_index = [tools["samtools"], "index", "-@", str(threads), str(merged_sorted)]
    run_command(cmd_index, index_log)
    state.mark_done(
        "merge_pooled_bam",
        outputs=[merged_sorted, merged_index],
        metadata=metadata,
    )
    return merged_sorted


def filter_vcf(
    vcf_path: pathlib.Path,
    tools: Dict[str, str],
    outdir: pathlib.Path,
    chrom: str,
    state: PipelineState,
) -> pathlib.Path:
    filtered_vcf = outdir / f"variants.{chrom}.snps.vcf.gz"
    filter_log = outdir / f"bcftools.filter.{chrom}.log"
    index_path = filtered_vcf.with_suffix(filtered_vcf.suffix + ".csi")
    metadata = {"chrom": chrom, "source_vcf": str(vcf_path)}
    if state.is_done("filter_vcf", chrom, expected_metadata=metadata):
        LOGGER.info("Filtered VCF for %s already exists at %s; skipping", chrom, filtered_vcf)
        return filtered_vcf
    prev_metadata = state.get_metadata("filter_vcf", chrom)
    forced = state.is_forced("filter_vcf", chrom)
    if (
        not forced
        and filtered_vcf.exists()
        and index_path.exists()
        and (not prev_metadata)
    ):
        LOGGER.info("Filtered VCF for %s found at %s; reusing", chrom, filtered_vcf)
        state.mark_done(
            "filter_vcf",
            chrom,
            outputs=[filtered_vcf, index_path],
            metadata=metadata,
        )
        return filtered_vcf
    if forced:
        LOGGER.info("Forcing regeneration of filtered VCF for %s", chrom)
    elif prev_metadata and prev_metadata != metadata:
        LOGGER.info("Metadata change detected for filtered VCF %s; regenerating", chrom)
    cmd = [
        tools["bcftools"],
        "view",
        "-r",
        chrom,
        "-v",
        "snps",
        "-m2",
        "-M2",
        "-Oz",
        "-o",
        str(filtered_vcf),
        str(vcf_path),
    ]
    run_command(cmd, filter_log)
    index_log = outdir / f"bcftools.index.{chrom}.log"
    cmd_index = [tools["bcftools"], "index", "-f", str(filtered_vcf)]
    run_command(cmd_index, index_log)
    state.mark_done(
        "filter_vcf",
        chrom,
        outputs=[filtered_vcf, index_path],
        metadata=metadata,
    )
    return filtered_vcf


def genotype_to_code(gt: Tuple[int, int]) -> float:
    if gt[0] == 0 and gt[1] == 0:
        return 0.0
    if gt[0] == 1 and gt[1] == 1:
        return 1.0
    if (gt[0] == 0 and gt[1] == 1) or (gt[0] == 1 and gt[1] == 0):
        return 0.5
    return float("nan")


def fetch_allele_counts(
    bam_path: pathlib.Path,
    vcf_path: pathlib.Path,
    standard_sample: str,
    min_depth: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, int, str, str]]]:
    bam = pysam.AlignmentFile(bam_path, "rb")
    vcf = pysam.VariantFile(vcf_path)
    if standard_sample not in vcf.header.samples:
        raise KeyError(f"Standard sample {standard_sample} not present in VCF")
    k_total: List[int] = []
    n_total: List[int] = []
    g_codes: List[float] = []
    snp_records: List[Tuple[str, int, str, str]] = []
    for rec in vcf.fetch():
        sample_data = rec.samples[standard_sample]
        gt = sample_data.get("GT")
        if gt is None or len(gt) != 2:
            continue
        gt_code = genotype_to_code(gt)
        if not (gt_code == gt_code):
            continue
        if len(rec.alts) != 1:
            continue
        chrom = rec.chrom
        pos = rec.pos
        ref = rec.ref.upper()
        alt = rec.alts[0].upper()
        column = None
        for pileup_col in bam.pileup(
            chrom,
            pos - 1,
            pos,
            truncate=True,
            stepper="nofilter",
            max_depth=1_000_000,
        ):
            if pileup_col.pos == pos - 1:
                column = pileup_col
                break
        if column is None:
            continue
        ref_count = 0
        alt_count = 0
        for pileup_read in column.pileups:
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
        total = ref_count + alt_count
        if total < min_depth:
            continue
        k_total.append(alt_count)
        n_total.append(total)
        g_codes.append(gt_code)
        snp_records.append((chrom, pos, ref, alt))
    bam.close()
    vcf.close()
    return np.array(k_total, dtype=np.int32), np.array(n_total, dtype=np.int32), np.array(g_codes, dtype=np.float32), snp_records


def save_counts(
    outdir: pathlib.Path,
    snps: Sequence[Tuple[str, int, str, str]],
    k_total: np.ndarray,
    n_total: np.ndarray,
    g_codes: np.ndarray,
) -> pathlib.Path:
    output = outdir / "allele_counts.tsv"
    with output.open("w", encoding="utf-8") as handle:
        handle.write("chrom\tpos\tref\talt\talt_depth\ttotal_depth\tstd_genotype\n")
        for (chrom, pos, ref, alt), k, n, g in zip(snps, k_total, n_total, g_codes):
            handle.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{k}\t{n}\t{g}\n")
    return output


def load_counts(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, int, str, str]]]:
    snps: List[Tuple[str, int, str, str]] = []
    alt_counts: List[int] = []
    total_counts: List[int] = []
    genotypes: List[float] = []
    with path.open("r", encoding="utf-8") as handle:
        header_read = False
        for line in handle:
            if not header_read:
                header_read = True
                continue
            line = line.strip()
            if not line:
                continue
            chrom, pos, ref, alt, alt_depth, total_depth, genotype = line.split("\t")
            snps.append((chrom, int(pos), ref, alt))
            alt_counts.append(int(alt_depth))
            total_counts.append(int(total_depth))
            genotypes.append(float(genotype))
    return (
        np.array(alt_counts, dtype=np.int32),
        np.array(total_counts, dtype=np.int32),
        np.array(genotypes, dtype=np.float32),
        snps,
    )


def estimate_purity(
    k_total: np.ndarray,
    n_total: np.ndarray,
    g_codes: np.ndarray,
    error_rate: float,
    min_depth: int,
    em_config: Dict[str, object],
) -> Dict[str, object]:
    c_unclipped = msim.error_correct_fraction_unclipped(k_total, n_total, error_rate)
    w_cov = msim.estimate_w_by_covariance(c_unclipped, g_codes, n_total, min_n=min_depth)
    c, g, wts = msim.mask_and_weights(c_unclipped, g_codes, n_total, min_n=min_depth)
    results = {
        "covariance_estimate": float(w_cov),
        "num_loci": int(len(c)),
    }
    if em_config.get("do_em", True) and len(c) >= 100:
        K_list = list(range(int(em_config["k_min"]), int(em_config["k_max"]) + 1))
        w_prior = None if not (w_cov == w_cov) else float(np.clip(w_cov, 1e-6, 1.0 - 1e-6))
        em_res = msim.model_select_k_em(
            c,
            g,
            wts,
            K_list=K_list,
            seed=int(em_config["seed"]),
            use_bic=bool(em_config["use_bic"]),
            use_icl=bool(em_config["use_icl"]),
            w_prior=w_prior,
            w_prior_tau=0.07,
            n_init=int(em_config["n_init"]),
            tau_schedule=tuple(em_config["tau_schedule"]),
            max_iter=int(em_config["max_iter"]),
            min_iter=int(em_config["min_iter"]),
        )
        results["em_result"] = em_res
        if em_res.get("success"):
            results["em_estimate"] = float(em_res.get("w", float("nan")))
            results["em_selected_K"] = int(em_res.get("K", -1))
    else:
        results["em_result"] = {"success": False, "reason": "insufficient_loci"}
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run pooled purity estimation on real data")
    parser.add_argument("--percentage-file", required=True, type=pathlib.Path)
    parser.add_argument("--cram-dir", required=True, type=pathlib.Path)
    parser.add_argument("--vcf", required=True, type=pathlib.Path)
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--chrom", default="chr1")
    parser.add_argument("--standard-sample", required=True)
    parser.add_argument("--target-depth", type=float, default=10.0, help="Total desired coverage (e.g. 10X)")
    parser.add_argument("--genome-size", type=float, default=4.5e9, help="Genome size in bases")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--min-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tool-config", type=pathlib.Path)
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--skip-em", action="store_true")
    parser.add_argument("--em-k-min", type=int, default=3)
    parser.add_argument("--em-k-max", type=int, default=9)
    parser.add_argument("--em-n-init", type=int, default=5)
    parser.add_argument("--em-max-iter", type=int, default=200)
    parser.add_argument("--em-min-iter", type=int, default=24)
    parser.add_argument("--em-tau", type=float, nargs="*", default=(1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05))
    parser.add_argument("--em-use-bic", action="store_true")
    parser.add_argument("--em-use-icl", action="store_true")
    parser.add_argument(
        "--force-step",
        action="append",
        default=[],
        metavar="STEP[:KEY]",
        help=(
            "Re-run a pipeline step even if its outputs exist. Can be provided multiple times. "
            "Use STEP for all keys or STEP:KEY for a specific sample/chromosome."
        ),
    )
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    args.outdir.mkdir(parents=True, exist_ok=True)
    tools = load_tool_paths(args.tool_config)

    state = PipelineState(
        args.outdir / "pipeline_state.json",
        forced=parse_force_steps(args.force_step),
    )

    percentages = parse_percentage_file(args.percentage_file)
    configs = collect_sample_configs(
        percentages,
        args.cram_dir,
        args.genome_size,
        args.target_depth,
        args.reference,
        args.outdir,
        tools,
        args.threads,
        args.seed,
        state,
    )
    for cfg in configs:
        subsample_cram(cfg, tools, args.reference, args.threads, args.seed, state)
    merged_bam = merge_bams(configs, tools, args.outdir, args.threads, state)
    filtered_vcf = filter_vcf(args.vcf, tools, args.outdir, args.chrom, state)
    counts_path = args.outdir / "allele_counts.tsv"
    counts_metadata = {
        "chrom": args.chrom,
        "standard_sample": args.standard_sample,
        "min_depth": int(args.min_depth),
        "merged_metadata": state.get_metadata("merge_pooled_bam"),
        "filtered_metadata": state.get_metadata("filter_vcf", args.chrom),
        "merged_bam": str(merged_bam),
        "filtered_vcf": str(filtered_vcf),
    }
    prev_counts_meta = state.get_metadata("allele_counts", args.chrom)
    forced_counts = state.is_forced("allele_counts", args.chrom)
    if state.is_done("allele_counts", args.chrom, expected_metadata=counts_metadata):
        LOGGER.info("Allele counts already present at %s; loading", counts_path)
        k_total, n_total, g_codes, snps = load_counts(counts_path)
    elif not forced_counts and counts_path.exists() and (not prev_counts_meta):
        LOGGER.info("Allele counts file detected at %s; reusing", counts_path)
        k_total, n_total, g_codes, snps = load_counts(counts_path)
        state.mark_done(
            "allele_counts",
            args.chrom,
            outputs=[counts_path],
            metadata=counts_metadata,
        )
    else:
        if forced_counts:
            LOGGER.info("Forcing regeneration of allele counts for %s", args.chrom)
        elif prev_counts_meta and prev_counts_meta != counts_metadata:
            LOGGER.info("Metadata change detected for allele counts on %s; recomputing", args.chrom)
        k_total, n_total, g_codes, snps = fetch_allele_counts(
            merged_bam,
            filtered_vcf,
            args.standard_sample,
            args.min_depth,
        )
        counts_path = save_counts(args.outdir, snps, k_total, n_total, g_codes)
        state.mark_done(
            "allele_counts",
            args.chrom,
            outputs=[counts_path],
            metadata=counts_metadata,
        )
    em_config = {
        "do_em": not args.skip_em,
        "k_min": args.em_k_min,
        "k_max": args.em_k_max,
        "seed": args.seed,
        "use_bic": args.em_use_bic,
        "use_icl": args.em_use_icl,
        "n_init": args.em_n_init,
        "tau_schedule": args.em_tau,
        "max_iter": args.em_max_iter,
        "min_iter": args.em_min_iter,
    }
    purity_results = estimate_purity(
        k_total,
        n_total,
        g_codes,
        args.error_rate,
        args.min_depth,
        em_config,
    )
    summary = {
        "percentage_file": str(args.percentage_file),
        "vcf": str(args.vcf),
        "reference": str(args.reference),
        "chrom": args.chrom,
        "standard_sample": args.standard_sample,
        "target_depth": args.target_depth,
        "genome_size": args.genome_size,
        "error_rate": args.error_rate,
        "min_depth": args.min_depth,
        "tools": tools,
        "samples": [
            {
                "name": cfg.name,
                "cram": str(cfg.cram_path),
                "fraction": cfg.fraction,
                "target_bases": cfg.target_bases,
                "total_bases": cfg.total_bases,
                "downsampled_bam": str(cfg.bam_path),
            }
            for cfg in configs
        ],
        "allele_counts": str(counts_path),
        "purity": purity_results,
        "pipeline_state": str(state.path),
    }
    summary_path = args.outdir / "purity_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Pipeline finished. Results written to %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
