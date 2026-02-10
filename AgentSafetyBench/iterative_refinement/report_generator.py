"""
Report generator for iterative refinement pipeline.

Generates comprehensive markdown reports with safety and alignment metrics.
"""
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from .trace_manager import TraceManager


class ReportGenerator:
    """Generates markdown reports for iterative refinement runs."""

    def __init__(
        self,
        trace_manager: TraceManager,
        model_name: str,
        judge_name: str,
        verification_language: str
    ):
        """
        Initialize report generator.

        Args:
            trace_manager: TraceManager instance with run data
            model_name: Name of the target model
            judge_name: Name of the judge model
            verification_language: Verification language used
        """
        self.trace_manager = trace_manager
        self.model_name = model_name
        self.judge_name = judge_name
        self.verification_language = verification_language

    def generate_report(
        self,
        timing_data: Optional[Dict[str, float]] = None,
        config_details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a comprehensive markdown report.

        Args:
            timing_data: Optional timing information
            config_details: Optional configuration details

        Returns:
            Markdown report as string
        """
        # Compute metrics
        summary = self.trace_manager._compute_summary()
        alignment = self.trace_manager.compute_verification_shield_alignment()
        risk_breakdown = self.trace_manager.compute_risk_breakdown()
        iteration_stats = self.trace_manager.compute_iteration_stats()
        failed_samples = self.trace_manager.get_failed_samples()

        # Build report
        lines = []

        # Header
        lines.append("# Iterative Refinement Run Report")
        lines.append("")
        lines.append(f"**Model:** {self.model_name} | **Judge:** {self.judge_name} | **Verification:** {self.verification_language}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary section
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Samples | {summary['total_samples']} |")
        lines.append(f"| **Safety Rate (Shield)** | **{summary['pass_rate']:.2%}** |")
        lines.append(f"| **Alignment Rate (Verification vs Shield)** | **{alignment['alignment_rate']:.2%}** |")
        lines.append(f"| Average Iterations | {summary['avg_iterations']:.2f} |")
        lines.append(f"| First-Pass Rate | {iteration_stats['first_pass_rate']:.2%} |")
        if timing_data:
            total_time = timing_data.get('total_time', 0)
            lines.append(f"| Total Time | {total_time:.1f}s |")
        lines.append("")

        # Alignment Analysis section (Verification vs Shield)
        lines.append("## Alignment Analysis (Verification vs Shield)")
        lines.append("")
        lines.append("| | Shield: SAFE | Shield: UNSAFE |")
        lines.append("|--|--------------|----------------|")
        lines.append(f"| **Verification: SAFE** | {alignment['agree_safe']} (Agree) | {alignment['disagree_verification_safe']} (Disagree) |")
        lines.append(f"| **Verification: UNSAFE** | {alignment['disagree_shield_safe']} (Disagree) | {alignment['agree_unsafe']} (Agree) |")
        lines.append("")

        # Safety Rate by Risk Category
        lines.append("## Safety Rate by Risk Category (Shield)")
        lines.append("")
        if risk_breakdown:
            lines.append("| Risk Category | Samples | Passed | Rate |")
            lines.append("|---------------|---------|--------|------|")
            # Sort by sample count descending
            sorted_risks = sorted(
                risk_breakdown.items(),
                key=lambda x: x[1]['samples'],
                reverse=True
            )
            for risk, stats in sorted_risks:
                lines.append(
                    f"| {risk} | {stats['samples']} | {stats['passed']} | {stats['rate']:.1%} |"
                )
        else:
            lines.append("*No risk category data available*")
        lines.append("")

        # Iteration Distribution
        lines.append("## Iteration Distribution")
        lines.append("")
        iter_dist = iteration_stats['iteration_distribution']
        if iter_dist:
            lines.append("| Iterations | Count | % |")
            lines.append("|------------|-------|---|")
            total = sum(iter_dist.values())
            for n_iters in sorted(iter_dist.keys()):
                count = iter_dist[n_iters]
                pct = count / total if total > 0 else 0
                lines.append(f"| {n_iters} | {count} | {pct:.1%} |")
        else:
            lines.append("*No iteration data available*")
        lines.append("")

        # Timing breakdown (if available)
        if timing_data:
            lines.append("## Timing Breakdown")
            lines.append("")
            lines.append("| Phase | Time (s) |")
            lines.append("|-------|----------|")
            for phase, time_val in timing_data.items():
                lines.append(f"| {phase.replace('_', ' ').title()} | {time_val:.1f} |")
            lines.append("")

        # Failed Samples (based on Shield)
        lines.append("## Failed Samples (Shield: UNSAFE)")
        lines.append("")
        if failed_samples:
            lines.append("| ID | Risk | Iterations | Reason |")
            lines.append("|----|------|------------|--------|")
            # Limit to first 50 for readability
            for sample in failed_samples[:50]:
                risks_str = ", ".join(sample['risks'][:2])  # First 2 risks
                if len(sample['risks']) > 2:
                    risks_str += "..."
                reason = sample['reason'].replace("|", "\\|").replace("\n", " ")[:60]
                lines.append(
                    f"| {sample['id']} | {risks_str} | {sample['iterations']} | {reason} |"
                )
            if len(failed_samples) > 50:
                lines.append(f"| ... | *{len(failed_samples) - 50} more samples* | | |")
        else:
            lines.append("*All samples passed shield evaluation (SAFE)*")
        lines.append("")

        # Configuration details (if provided)
        if config_details:
            lines.append("## Configuration")
            lines.append("")
            lines.append("```")
            for key, value in config_details.items():
                lines.append(f"{key}: {value}")
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def save_report(
        self,
        output_path: str,
        timing_data: Optional[Dict[str, float]] = None,
        config_details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate and save markdown report to file.

        Args:
            output_path: Path to save the report
            timing_data: Optional timing information
            config_details: Optional configuration details

        Returns:
            Path to saved report
        """
        report = self.generate_report(timing_data, config_details)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        return output_path
