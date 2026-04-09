import argparse
import csv
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--summary-csv", required=True)
    p.add_argument("--report-md", required=True)
    return p.parse_args()


def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


def main():
    args = parse_args()

    grouped = defaultdict(list)
    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["version"], int(row["n"]))
            grouped[key].append(row)

    summary_rows = []
    for (version, n), rows in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        end_vals = [float(r["end_to_end_s"]) for r in rows]
        comp_vals = [float(r["compute_s"]) for r in rows]
        gflops_vals = [float(r["gflops_compute"]) for r in rows]
        max_err = max(float(r["max_abs_error"]) for r in rows)
        summary_rows.append(
            {
                "version": version,
                "n": n,
                "repeats": len(rows),
                "avg_end_to_end_s": avg(end_vals),
                "avg_compute_s": avg(comp_vals),
                "avg_gflops_compute": avg(gflops_vals),
                "max_abs_error": max_err,
            }
        )

    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "version",
            "n",
            "repeats",
            "avg_end_to_end_s",
            "avg_compute_s",
            "avg_gflops_compute",
            "max_abs_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    versions_by_n = defaultdict(list)
    for row in summary_rows:
        versions_by_n[row["n"]].append(row)

    lines = []
    lines.append("# Polaris GEMM 5-Way Comparison")
    lines.append("")
    lines.append("Averages are over measured repeats (warmup excluded).")
    lines.append("")

    for n in sorted(versions_by_n.keys()):
        lines.append(f"## n = {n}")
        lines.append("")
        lines.append("| Version | Repeats | Avg End-to-End (s) | Avg Compute (s) | Avg Compute GFLOP/s | Max Abs Error |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in sorted(versions_by_n[n], key=lambda x: x["version"]):
            lines.append(
                f"| {row['version']} | {row['repeats']} | "
                f"{row['avg_end_to_end_s']:.6f} | {row['avg_compute_s']:.6f} | "
                f"{row['avg_gflops_compute']:.3f} | {row['max_abs_error']:.3e} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `avg_end_to_end_s` includes transfer + compute + copy-back when applicable.")
    lines.append("- `avg_compute_s` corresponds to the timed compute region only.")
    lines.append("- `max_abs_error` is max over repeats for each (version, n).")

    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
