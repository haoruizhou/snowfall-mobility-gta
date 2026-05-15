# %% [markdown]
# # Paper Figures
#
# Generates all figures and Table 1 statistics reported in:
# **"Mobility Responses to Snowfall in the Greater Toronto Area"**
# Zhou & Long, Western University
#
# | Figure | Content                                     | Events              |
# |--------|---------------------------------------------|---------------------|
# | Fig 1  | Scatter: trip diff vs daily snowfall        | ≥1 cm and ≥5 cm     |
# | Fig 2  | VKT 4-panel                                 | all ≥5 cm days      |
# | Fig 3  | 3-point temporal: before/during/after       | independent ≥5 cm   |
# | Tbl 1  | Paired t-tests ±1d and ±2d                  | independent ≥5 cm   |
#
# "Temporally independent" events: sessions where neither the day before
# the session start nor the day after the session end records ≥5 cm snow.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import polars as pl
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
plt.ioff()  # Turn off interactive mode
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy import stats as scipy_stats

from utils import (
    load_data, compute_daily_deviation, compute_window_deviation,
    collect_session_events, build_daily_snow_map, filter_independent_events,
    paired_ttest, brown_forsythe,
)
from config import (
    MIN_SNOW_CM, ANY_SNOW_CM, VKT_COLS, VKT_PARQUET,
    apply_plot_style, style_axes,
    BG, FG, FG_SUBTLE, FG_TITLE, SPINE,
    C_MEAN, C_ZONE_PRE, C_ZONE_DURING, C_ZONE_POST,
    C_BLUE, C_ORANGE, C_VERMILLION, C_GREEN, PALETTE,
)

apply_plot_style()

OUT_DIR = Path("paper_outputs")
OUT_DIR.mkdir(exist_ok=True)

# %%
print("Loading data …")
df, event_ranges = load_data()
daily_snow_map   = build_daily_snow_map(df)

print("\nBuilding session_event_df (all sessions with all days ≥5 cm) …")
session_df = collect_session_events(event_ranges, df, min_snow_cm=MIN_SNOW_CM, lead_days=1)

print("\nFiltering to temporally independent 1-day events …")
indep_df = filter_independent_events(session_df, daily_snow_map, threshold=MIN_SNOW_CM)
N_INDEP  = indep_df.height

print(f"\n  All qualifying sessions  : {session_df.height}")
print(f"  Temporally independent   : {N_INDEP}")

# %% [markdown]
# ---
# ## Figure 1 — Scatter: Trip Volume Difference vs Daily Snowfall

# %%
day_summaries = []
for row in event_ranges.iter_rows(named=True):
    day_sub = df.filter(
        (pl.col("bucket_start") >= row["event_start"]) &
        (pl.col("bucket_start") <  row["event_end"])
    )
    if day_sub.height == 0:
        continue
    actual   = int(day_sub["count"].cast(pl.Int64).sum())
    baseline = float(day_sub["baseline"].sum())
    day_summaries.append({"daily_snow_cm": row["daily_snow_cm"], "difference": actual - baseline})

day_df_all = pl.DataFrame(day_summaries)
days_ge1   = day_df_all.filter(pl.col("daily_snow_cm") >= ANY_SNOW_CM)
days_ge5   = day_df_all.filter(pl.col("daily_snow_cm") >= MIN_SNOW_CM)
print(f"\nFig 1 — days ≥{ANY_SNOW_CM} cm: {days_ge1.height}  |  ≥{MIN_SNOW_CM} cm: {days_ge5.height}")

fig1, ax1 = plt.subplots(figsize=(15, 10))

def _scatter(ax, x, y, color, marker, label, zorder):
    ax.scatter(x, y, alpha=0.75, s=130, color=color, marker=marker,
               edgecolors="none", label=label, zorder=zorder)

_scatter(ax1, days_ge1["daily_snow_cm"].to_numpy(), days_ge1["difference"].to_numpy(),
         C_BLUE, "o", f"Snow ≥{ANY_SNOW_CM} cm (n={days_ge1.height})", 4)
_scatter(ax1, days_ge5["daily_snow_cm"].to_numpy(), days_ge5["difference"].to_numpy(),
         C_VERMILLION, "D", f"Snow ≥{MIN_SNOW_CM} cm (n={days_ge5.height})", 5)

ax1.axhline(0, color="#9ca3af", linewidth=1.25, alpha=0.65)

fig1_stats = {}
for label, ds, color, key in [
    (f"≥{ANY_SNOW_CM} cm", days_ge1, C_BLUE,       "ge1"),
    (f"≥{MIN_SNOW_CM} cm", days_ge5, C_VERMILLION,  "ge5"),
]:
    x_arr = ds["daily_snow_cm"].to_numpy()
    y_arr = ds["difference"].to_numpy()
    if x_arr.size > 1:
        slope, intercept, r, p, se = scipy_stats.linregress(x_arr, y_arr)
        xline = np.linspace(x_arr.min(), x_arr.max(), 100)
        ax1.plot(xline, slope * xline + intercept, color=color, linewidth=2.75, alpha=0.85,
                 label=f"{label} trend (r={r:.3f}, p={p:.3f})", zorder=3)
        fig1_stats[key] = {"r": r, "p": p, "slope": slope, "n": len(x_arr)}

ax1.set_xlabel("Daily Snowfall (cm)", fontsize=19)
ax1.set_ylabel("Trip Volume Difference (observed − baseline)", fontsize=19)
ax1.set_title("Figure 1. Trip Volume Difference vs Daily Snowfall",
              fontsize=24, fontweight="medium", pad=18, color=FG_TITLE)
style_axes(ax1); ax1.tick_params(labelsize=18)
ax1.legend(fontsize=15, frameon=False, labelcolor=FG_SUBTLE)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_scatter.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

print("\n  Fig 1 regression:")
for key, s in fig1_stats.items():
    sig = "***" if s["p"] < 0.001 else "**" if s["p"] < 0.01 else "*" if s["p"] < 0.05 else "ns"
    print(f"    {key}: r={s['r']:.3f}  p={s['p']:.3f}  slope={s['slope']:.1f} trips/cm  {sig}")

# %% [markdown]
# ---
# ## Figure 2 — VKT 4-Panel

# %%
df_vkt = pl.read_parquet(str(VKT_PARQUET), columns=VKT_COLS)

ROAD_GROUP_MAP   = {"Motorway":"Highway","Trunk":"Highway",
                    "Primary":"Main Roads","Secondary":"Main Roads","Tertiary":"Main Roads",
                    "Residential":"Local Roads","Unclassified":"Local Roads"}
ROAD_GROUP_ORDER = ["Highway","Main Roads","Local Roads"]
ROAD_COLS_VKT    = ["motorway_km","trunk_km","primary_km","secondary_km",
                    "tertiary_km","residential_km","unclassified_km"]
DIST_ORDER       = ["0-2km","2-5km","5-10km","10-20km","20+km"]

def _vkt_analysis(df_in):
    vkt_exprs = []
    for rc in ROAD_COLS_VKT:
        vkt_exprs += [
            (pl.col("count")    * pl.col(rc)).alias(f"av_{rc}"),
            (pl.col("baseline") * pl.col(rc)).alias(f"bv_{rc}"),
        ]
    vkt_exprs += [
        (pl.col("count")    * pl.col("total_distance_km")).alias("av_total"),
        (pl.col("baseline") * pl.col("total_distance_km")).alias("bv_total"),
    ]
    df_v = df_in.with_columns(vkt_exprs).with_columns([
        pl.when(pl.col("total_distance_km") < 2).then(pl.lit("0-2km"))
        .when(pl.col("total_distance_km") < 5).then(pl.lit("2-5km"))
        .when(pl.col("total_distance_km") < 10).then(pl.lit("5-10km"))
        .when(pl.col("total_distance_km") < 20).then(pl.lit("10-20km"))
        .otherwise(pl.lit("20+km")).alias("distance_bin")
    ])
    dist = (
        df_v.group_by("distance_bin").agg([
            pl.col("count").sum().alias("actual_trips"),
            pl.col("baseline").sum().alias("baseline_trips"),
            pl.col("av_total").sum().alias("actual_vkt"),
            pl.col("bv_total").sum().alias("baseline_vkt"),
            pl.col("total_distance_km").mean().alias("avg_dist"),
        ])
        .with_columns([
            ((pl.col("actual_trips") - pl.col("baseline_trips")) / pl.col("baseline_trips") * 100).alias("trip_pct"),
            ((pl.col("actual_vkt")   - pl.col("baseline_vkt"))   / pl.col("baseline_vkt")   * 100).alias("vkt_pct"),
        ])
        .sort("avg_dist")
    )
    road_rows = []
    for rc in ROAD_COLS_VKT:
        rt = rc.replace("_km","").title()
        road_rows.append(
            df_v.select([pl.col("count").sum().alias("at"), pl.col("baseline").sum().alias("bt"),
                         pl.col(f"av_{rc}").sum().alias("av"), pl.col(f"bv_{rc}").sum().alias("bv")])
            .with_columns(pl.lit(rt).alias("road_type"))
        )
    road = (
        pl.concat(road_rows)
        .with_columns(((pl.col("av") - pl.col("bv")) / pl.col("bv") * 100).alias("vkt_pct"))
        .sort("av", descending=True)
    )
    return dist, road

print(f"\nFig 2 — processing individual days ≥{MIN_SNOW_CM} cm …")
ev_filtered = event_ranges.filter(pl.col("daily_snow_cm") >= MIN_SNOW_CM)
ev_results  = []
for row in ev_filtered.iter_rows(named=True):
    df_evt = df_vkt.filter(
        (pl.col("bucket_start") >= row["event_start"]) &
        (pl.col("bucket_start") <  row["event_end"])   &
        pl.col("baseline").is_not_null()
    )
    if df_evt.height == 0: continue
    d, r = _vkt_analysis(df_evt)
    ev_results.append({"distance": d, "road": r})

N_VKT = len(ev_results)
print(f"  Events with data: {N_VKT}")

all_dist = pd.concat([r["distance"].to_pandas() for r in ev_results], ignore_index=True)
dist_mean = (all_dist.groupby("distance_bin")
    .agg({c:"mean" for c in ["actual_trips","baseline_trips","actual_vkt","baseline_vkt","trip_pct","vkt_pct"]})
    .reset_index())
dist_mean = dist_mean.set_index("distance_bin").reindex(DIST_ORDER).reset_index().fillna(0)

all_road  = pd.concat([r["road"].to_pandas() for r in ev_results], ignore_index=True)
road_mean = (all_road.groupby("road_type").agg({"av":"mean","bv":"mean","vkt_pct":"mean"}).reset_index()
             .rename(columns={"av":"actual_vkt","bv":"baseline_vkt"}))
road_mean["road_group"] = road_mean["road_type"].map(ROAD_GROUP_MAP)
road_simple = (road_mean.groupby("road_group")
    .agg({"actual_vkt":"sum","baseline_vkt":"sum"}).reset_index())
road_simple["vkt_pct"] = (road_simple["actual_vkt"] - road_simple["baseline_vkt"]) / road_simple["baseline_vkt"] * 100
road_simple = road_simple.set_index("road_group").reindex(ROAD_GROUP_ORDER).reset_index()

def _millions(x, _):
    if abs(x)>=1e6: return f"{x/1e6:.1f}M"
    if abs(x)>=1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"

C_ACT = C_BLUE; C_BASE = "#D3D3D3"; C_VKT = C_ORANGE; C_NEG = C_VERMILLION; C_POS = C_GREEN
fig2 = plt.figure(figsize=(19,15))
gs2  = mgs.GridSpec(2,2,hspace=0.32,wspace=0.25)
fig2.suptitle(f"Figure 2. Trip Volume and VKT — Mean Across {N_VKT} Days (≥{MIN_SNOW_CM} cm)",
              fontsize=26, fontweight="medium", color=FG_TITLE, y=0.97)
x = np.arange(len(DIST_ORDER)); xr = np.arange(len(ROAD_GROUP_ORDER)); w=0.35

ax2a = fig2.add_subplot(gs2[0,0])
ax2a.bar(x-w/2, dist_mean["actual_trips"],   w, color=C_ACT,  alpha=0.85, label="Actual")
ax2a.bar(x+w/2, dist_mean["baseline_trips"], w, color=C_BASE, alpha=0.85, label="Baseline")
ax2a.set_xticks(x); ax2a.set_xticklabels(DIST_ORDER, fontsize=16)
ax2a.set_ylabel("Trip Volume", fontsize=16)
ax2a.set_title("(a) Trip Volume by Distance", fontsize=18, fontweight="medium")
ax2a.legend(fontsize=13, frameon=False); ax2a.yaxis.set_major_formatter(FuncFormatter(_millions))
style_axes(ax2a); ax2a.tick_params(labelsize=16)

ax2b = fig2.add_subplot(gs2[0,1])
ax2b.bar(xr-w/2, road_simple["actual_vkt"],   w, color=C_ACT,  alpha=0.85, label="Actual")
ax2b.bar(xr+w/2, road_simple["baseline_vkt"], w, color=C_BASE, alpha=0.85, label="Baseline")
ax2b.set_xticks(xr); ax2b.set_xticklabels([f"{g}*" for g in ROAD_GROUP_ORDER], fontsize=16)
ax2b.set_ylabel("VKT", fontsize=16)
ax2b.set_title("(b) VKT by Road Type", fontsize=18, fontweight="medium")
ax2b.legend(fontsize=13, frameon=False); ax2b.yaxis.set_major_formatter(FuncFormatter(_millions))
style_axes(ax2b); ax2b.tick_params(labelsize=16)

ax2c = fig2.add_subplot(gs2[1,0])
bc1 = ax2c.bar(x-w/2, dist_mean["trip_pct"], w, color=C_ACT, alpha=0.85, label="Trip Volume")
bc2 = ax2c.bar(x+w/2, dist_mean["vkt_pct"],  w, color=C_VKT, alpha=0.85, label="VKT")
for bars in [zip(bc1, dist_mean["trip_pct"]), zip(bc2, dist_mean["vkt_pct"])]:
    for bar, val in bars:
        ax2c.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{val:+.1f}%",
                  ha="center", va="bottom" if val>=0 else "top", fontsize=11, color=FG_SUBTLE)
ax2c.axhline(0, color="#9ca3af", linewidth=1.25, alpha=0.65)
ax2c.set_xticks(x); ax2c.set_xticklabels(DIST_ORDER, fontsize=16)
ax2c.set_ylabel("Change from Baseline (%)", fontsize=16)
ax2c.set_title("(c) % Change by Distance", fontsize=18, fontweight="medium")
ax2c.legend(fontsize=13, frameon=False); style_axes(ax2c); ax2c.tick_params(labelsize=16)

ax2d = fig2.add_subplot(gs2[1,1])
bar_colors = [C_NEG if v<0 else C_POS for v in road_simple["vkt_pct"]]
bd = ax2d.bar(xr, road_simple["vkt_pct"], width=0.5, color=bar_colors, alpha=0.85)
for bar, ch in zip(bd, road_simple["vkt_pct"]):
    ax2d.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{ch:+.1f}%",
              ha="center", va="bottom" if ch>=0 else "top", fontsize=14, color=FG_SUBTLE)
ax2d.axhline(0, color="#9ca3af", linewidth=1.25, alpha=0.65)
ax2d.set_xticks(xr); ax2d.set_xticklabels([f"{g}*" for g in ROAD_GROUP_ORDER], fontsize=16)
ax2d.set_ylabel("VKT Deviation (%)", fontsize=16)
ax2d.set_title("(d) VKT % Change by Road Type", fontsize=18, fontweight="medium")
style_axes(ax2d); ax2d.tick_params(labelsize=16)

plt.savefig(OUT_DIR/"fig2_vkt.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
print("  Saved: fig2_vkt.png")

# %% [markdown]
# ---
# ## Figure 3 — 3-Point Temporal Plot (temporally independent events)

# %%
print(f"\nFig 3 — n={N_INDEP} temporally independent events")
if N_INDEP == 0:
    print("  WARNING: no independent events found; check daily_snow_map and threshold.")
else:
    pre_i    = indep_df["pre_dev"].to_numpy()
    during_i = indep_df["during_dev"].to_numpy()
    post_i   = indep_df["post_dev"].to_numpy()
    n_days_vals = sorted(indep_df["n_days"].unique().to_list())
    day_colors  = {nd: PALETTE[i % len(PALETTE)] for i, nd in enumerate(n_days_vals)}

    fig3, ax3 = plt.subplots(figsize=(13, 9))

    # Boxplots
    for xi, vals, color in [(0, pre_i, C_ZONE_PRE), (1, during_i, C_ZONE_DURING), (2, post_i, C_ZONE_POST)]:
        ax3.boxplot(
            vals, positions=[xi], widths=0.42, patch_artist=True, zorder=1, showfliers=True,
            boxprops=dict(facecolor=color, alpha=0.50, edgecolor=SPINE, linewidth=1.5),
            medianprops=dict(color=FG_SUBTLE, linewidth=2.2, solid_capstyle="round"),
            whiskerprops=dict(color=SPINE, linewidth=1.2),
            capprops=dict(color=SPINE, linewidth=1.2),
            flierprops=dict(marker="o", markerfacecolor=SPINE, markeredgecolor="none",
                            alpha=0.40, markersize=4),
        )

    # Individual event lines (colored by session duration)
    legend_handles = []
    seen = set()
    for row in indep_df.iter_rows(named=True):
        nd = row["n_days"]
        color = day_colors[nd]
        ys = [row["pre_dev"], row["during_dev"], row["post_dev"]]
        ax3.plot([0, 1, 2], ys, color=color, alpha=0.38, linewidth=1.3, zorder=2)
        ax3.scatter([0, 1, 2], ys, color=color, s=40, alpha=0.65, zorder=3, edgecolors="none")
        if nd not in seen:
            cnt = indep_df.filter(pl.col("n_days") == nd).height
            legend_handles.append(
                Line2D([0], [0], color=color, marker="o", linewidth=1.3, markersize=7,
                       label=f"{nd}-day event (n={cnt})")
            )
            seen.add(nd)

    # Mean diamonds
    for xi, vals in [(0, pre_i), (1, during_i), (2, post_i)]:
        ax3.scatter([xi], [np.mean(vals)], color=C_MEAN, s=210, zorder=6,
                    marker="D", edgecolors="white", linewidths=1.5)

    ax3.axhline(0, color="#9ca3af", linewidth=1.25, alpha=0.55, zorder=2)

    legend_handles.insert(0, Line2D(
        [0], [0], marker="D", color=C_MEAN, markersize=11, linewidth=0,
        markeredgecolor="white", markeredgewidth=1.5, label="Mean"
    ))
    ax3.set_title(
        f"Figure 3. Mobility Deviation Before / During / After Snowfall\n"
        f"Temporally independent events  n={N_INDEP}  (≥{MIN_SNOW_CM} cm, no adjacent ≥5 cm day)",
        fontsize=20, fontweight="medium", pad=16, color=FG_TITLE,
    )
    ax3.set_ylabel("Mobility Deviation from Baseline", fontsize=16, color=FG_SUBTLE)
    ax3.tick_params(axis="y", labelsize=16)
    ax3.set_xticks([0,1,2])
    ax3.set_xticklabels(["Before\n(−1 day)", "During\n(event day)", "After\n(+1 day)"], fontsize=16)
    ax3.set_xlim(-0.65, 2.65)
    style_axes(ax3)
    ax3.legend(handles=legend_handles, loc="lower left", fontsize=13,
               frameon=True, labelcolor=FG_SUBTLE, facecolor="white",
               edgecolor=SPINE, framealpha=0.95).set_zorder(10)
    plt.tight_layout(pad=2.0)
    plt.savefig(OUT_DIR/"fig3_temporal.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    print("  Saved: fig3_temporal.png")

# %% [markdown]
# ---
# ## Table 1 — Paired t-tests (±1d and ±2d windows)
#
# The paper reports ±1-day and ±2-day windows for the independent events.
# For window > 1 day, pre and post are means over the W days before/after.

# %%
print(f"\n{'='*80}")
print(f"TABLE 1 — Temporally independent events  (n = {N_INDEP})")
print(f"{'='*80}")
print(f"{'Comparison':<20} {'Window':>7} {'MobDev pp':>10} {'t':>7} {'p':>8} {'d':>7}  Sig")
print("-"*80)

WINDOWS = [1, 2]

for W in WINDOWS:
    # Recompute pre/post with the given window size
    rows_w = []
    for row in indep_df.iter_rows(named=True):
        pre_w  = compute_window_deviation(df, row["session_start"], W, "pre")
        post_w = compute_window_deviation(df, row["session_end"],   W, "post")
        dur_w  = row["during_dev"]   # always the same (event day average)
        if pre_w is not None and post_w is not None:
            rows_w.append({"pre": pre_w, "during": dur_w, "post": post_w})

    if not rows_w:
        print(f"  No data for ±{W}d window")
        continue

    pre_w    = np.array([r["pre"]    for r in rows_w])
    during_w = np.array([r["during"] for r in rows_w])
    post_w   = np.array([r["post"]   for r in rows_w])

    for label, a, b in [
        ("Pre vs During",  pre_w,    during_w),
        ("During vs Post", during_w, post_w),
        ("Pre vs Post",    pre_w,    post_w),
    ]:
        r = paired_ttest(a, b)
        print(f"  {label:<20} ±{W}d    {r['mean_diff_pct']:>+8.2f}pp  "
              f"{r['t']:>+6.3f}  {r['p']:>8.4f}  {r['d']:>+6.3f}  {r['sig']}")
    print()

# %%
if N_INDEP > 0:
    print(f"{'='*80}")
    print(f"Brown-Forsythe variance tests  (±1d independent events,  n={N_INDEP})")
    print(f"{'='*80}")
    pre1    = indep_df["pre_dev"].to_numpy()
    during1 = indep_df["during_dev"].to_numpy()
    post1   = indep_df["post_dev"].to_numpy()

    bf_all = brown_forsythe(pre1, during1, post1)
    bf_pp  = brown_forsythe(pre1, post1)
    print(f"  All zones (pre/during/post): stat={bf_all['stat']:.4f}  p={bf_all['p']:.4f}  "
          f"{'significant' if bf_all['significant'] else 'ns'}")
    print(f"  Pre vs Post:                stat={bf_pp['stat']:.4f}  p={bf_pp['p']:.4f}  "
          f"{'significant' if bf_pp['significant'] else 'ns'}")

# %%
print(f"\n{'='*80}")
print("PAPER FIGURE SUMMARY")
print(f"{'='*80}")
n_ge1 = event_ranges.filter(pl.col("daily_snow_cm") >= ANY_SNOW_CM).height
n_ge5 = event_ranges.filter(pl.col("daily_snow_cm") >= MIN_SNOW_CM).height
print(f"  Days ≥{ANY_SNOW_CM} cm (Fig 1 scatter, series 1)  : {n_ge1}")
print(f"  Days ≥{MIN_SNOW_CM} cm (Fig 1 scatter + Fig 2 VKT): {n_ge5}")
print(f"  All sessions ≥{MIN_SNOW_CM} cm                    : {session_df.height}")
print(f"  Independent events (Fig 3 + Table 1)             : {N_INDEP}")
print(f"\n  Zone means (independent events):")
if N_INDEP > 0:
    print(f"    pre    : {np.mean(pre1)*100:+.2f}%  (s.d. = {np.std(pre1, ddof=1)*100:.2f} pp)")
    print(f"    during : {np.mean(during1)*100:+.2f}%  (s.d. = {np.std(during1, ddof=1)*100:.2f} pp)")
    print(f"    post   : {np.mean(post1)*100:+.2f}%  (s.d. = {np.std(post1, ddof=1)*100:.2f} pp)")
print(f"\nAll figures saved to: {OUT_DIR}/")
