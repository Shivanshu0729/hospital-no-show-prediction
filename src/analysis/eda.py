from typing import Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config import FIGURES_DIR, WEEKDAYS, PROCESSED_DATA_PATH


def run_exploratory_data_analysis(
    df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})

    total_records = len(df)
    total_attended = int((df["no_show"] == 0).sum())
    total_missed = int((df["no_show"] == 1).sum())
    noshow_rate = float(df["no_show"].mean())

    missing_counts = df.isnull().sum().to_dict()
    duplicates_count = int(df.duplicated().sum())

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        x="no_show",
        data=df,
        ax=ax,
        palette=["#2b5c8f", "#d95f02"],
        hue="no_show",
        legend=False,
    )
    ax.set_title("Attendance vs No-Show Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Status (0 = Attended, 1 = Missed)", fontsize=10)
    ax.set_ylabel("Appointment Count", fontsize=10)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height):,} ({height/total_records*100:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )
    plt.tight_layout()
    plt.savefig(output_dir / "attendance_distribution.png", dpi=200)
    plt.close()

    weekday_order = [w for w in WEEKDAYS if w in df["appointment_weekday"].unique()]
    fig, ax = plt.subplots(figsize=(8, 4))
    weekday_summary = (
        df.groupby("appointment_weekday")["no_show"]
        .mean()
        .reindex(weekday_order)
        .reset_index()
    )
    sns.barplot(
        x="appointment_weekday",
        y="no_show",
        data=weekday_summary,
        ax=ax,
        palette="Blues_r",
        hue="appointment_weekday",
        legend=False,
    )
    ax.set_title("No-Show Rate by Appointment Weekday", fontsize=12, fontweight="bold")
    ax.set_xlabel("Weekday", fontsize=10)
    ax.set_ylabel("No-Show Rate", fontsize=10)
    ax.set_ylim(0, max(0.4, weekday_summary["no_show"].max() * 1.25))
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.annotate(
                f"{h*100:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )
    plt.tight_layout()
    plt.savefig(output_dir / "noshow_by_weekday.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    hour_summary = (
        df.groupby("appointment_hour")["no_show"]
        .mean()
        .reset_index()
        .sort_values(by="appointment_hour")
    )
    sns.barplot(
        x="appointment_hour",
        y="no_show",
        data=hour_summary,
        ax=ax,
        palette="crest",
        hue="appointment_hour",
        legend=False,
    )
    ax.set_title("No-Show Rate by Appointment Hour", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day (24-hour)", fontsize=10)
    ax.set_ylabel("No-Show Rate", fontsize=10)
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.annotate(
                f"{h*100:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
    plt.tight_layout()
    plt.savefig(output_dir / "noshow_by_hour.png", dpi=200)
    plt.close()

    df_lt = df.copy()
    bins = [-1, 0, 3, 7, 14, 30, 100]
    labels = ["Same Day (0)", "1-3 Days", "4-7 Days", "8-14 Days", "15-30 Days", "30+ Days"]
    df_lt["lead_time_group"] = pd.cut(df_lt["lead_time_days"], bins=bins, labels=labels)
    lt_summary = (
        df_lt.groupby("lead_time_group", observed=False)["no_show"]
        .agg(["mean", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        x="lead_time_group",
        y="mean",
        data=lt_summary,
        ax=ax,
        palette="viridis",
        hue="lead_time_group",
        legend=False,
    )
    ax.set_title("No-Show Rate by Lead Time Category", fontsize=12, fontweight="bold")
    ax.set_xlabel("Lead Time Group", fontsize=10)
    ax.set_ylabel("No-Show Rate", fontsize=10)
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.annotate(
                f"{h*100:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )
    plt.tight_layout()
    plt.savefig(output_dir / "noshow_by_lead_time.png", dpi=200)
    plt.close()

    channel_summary = (
        df.groupby("booking_channel")["no_show"]
        .agg(["mean", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x="booking_channel",
        y="mean",
        data=channel_summary,
        ax=ax,
        palette="Set2",
        hue="booking_channel",
        legend=False,
    )
    ax.set_title("No-Show Rate by Booking Channel", fontsize=12, fontweight="bold")
    ax.set_xlabel("Booking Channel", fontsize=10)
    ax.set_ylabel("No-Show Rate", fontsize=10)
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.annotate(
                f"{h*100:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )
    plt.tight_layout()
    plt.savefig(output_dir / "noshow_by_booking_channel.png", dpi=200)
    plt.close()

    df_hist = df.copy()
    df_hist["history_bucket"] = "0 Previous Missed"
    df_hist.loc[df_hist["previous_no_shows"] == 1, "history_bucket"] = "1 Previous Missed"
    df_hist.loc[df_hist["previous_no_shows"] >= 2, "history_bucket"] = "2+ Previous Missed"
    bucket_order = ["0 Previous Missed", "1 Previous Missed", "2+ Previous Missed"]

    hist_summary = (
        df_hist.groupby("history_bucket")["no_show"]
        .agg(["mean", "count"])
        .reindex(bucket_order)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x="history_bucket",
        y="mean",
        data=hist_summary,
        ax=ax,
        palette="magma",
        hue="history_bucket",
        legend=False,
    )
    ax.set_title("No-Show Rate by Historical Missed Appointments", fontsize=12, fontweight="bold")
    ax.set_xlabel("Historical No-Show Status", fontsize=10)
    ax.set_ylabel("Current Appointment No-Show Rate", fontsize=10)
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.annotate(
                f"{h*100:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )
    plt.tight_layout()
    plt.savefig(output_dir / "previous_noshow_analysis.png", dpi=200)
    plt.close()

    num_cols = ["lead_time_days", "appointment_hour", "previous_appointments", "previous_no_shows", "no_show"]
    corr_matrix = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Correlation Heatmap of Numerical Features", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    summary_metrics = {
        "total_records": total_records,
        "total_attended": total_attended,
        "total_missed": total_missed,
        "noshow_rate": round(noshow_rate, 4),
        "lead_time_median": float(df["lead_time_days"].median()),
        "lead_time_mean": float(df["lead_time_days"].mean()),
        "missing_counts": missing_counts,
        "duplicates_count": duplicates_count,
        "weekday_noshow_rates": weekday_summary.set_index("appointment_weekday")["no_show"].round(4).to_dict(),
        "channel_noshow_rates": channel_summary.set_index("booking_channel")["mean"].round(4).to_dict(),
        "history_noshow_rates": hist_summary.set_index("history_bucket")["mean"].round(4).to_dict(),
    }

    return summary_metrics


if __name__ == "__main__":
    if PROCESSED_DATA_PATH.exists():
        data = pd.read_csv(PROCESSED_DATA_PATH)
        run_exploratory_data_analysis(data)
