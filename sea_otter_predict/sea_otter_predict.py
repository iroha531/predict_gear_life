"""
sea_otter_predict.py
====================
ラッコ（Enhydra lutris）個体数推定モデル
- 仮想 CSV データ生成
- 階層ベイズ状態空間モデル（PyMC）
- MCMC サンプリング & 事後分布可視化

[修正履歴]
----------
v1 (初版):
  obs_cap を Binomial(n=pm.math.round(N_obs).astype(int), p=p_cap) で定義。
  → ジッター処理 (_init_jitter) で int キャストが原因の
     AttributeError: 'Scratchpad' object has no attribute 'ufunc' が発生。

v2:
  obs_cap を Poisson(mu=N_obs * p_cap) に変更したが、
  pm.Normal ループ + pm.math.stack + 2D AdvancedSubtensor の構造が残り同エラー再発。

v3:
  ① 状態空間を GaussianRandomWalk に置き換え (scan ベースでジッターと適合)
  ② 2D AdvancedSubtensor → reshape(-1,) + 1D インデックスに変更
  ③ obs_cap は Poisson 近似を維持
  → AttributeError 解消・サンプリング正常動作を確認済み。

v4 (本版):
  出力画像の日本語文字化け修正。
  Noto Sans CJK JP フォントを FontProperties で明示指定し、
  すべての日本語テキスト (タイトル・軸ラベル・凡例・テキスト) に適用。
"""
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from pathlib import Path

# ─────────────────────────────────────────────
# 0. シード固定 & 日本語フォント設定
# ─────────────────────────────────────────────
RNG = np.random.default_rng(42)

# システムにある Noto Sans CJK JP を使用（文字化け対策）
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
JP_FONT = fm.FontProperties(fname=_FONT_PATH)

def _jp(text: str, size: float | None = None) -> dict:
    """日本語テキスト用の fontproperties kwargs を返す便利関数。"""
    prop = fm.FontProperties(fname=_FONT_PATH, size=size) if size else JP_FONT
    return {"fontproperties": prop}


# ─────────────────────────────────────────────
# 1. 仮想データ生成
# ─────────────────────────────────────────────
def generate_virtual_data(csv_path: str = "sea_otter_data.csv") -> pd.DataFrame:
    """
    複数調査地点・複数年の仮想捕獲再捕獲データを生成して CSV に保存する。
    """
    sites = list("ABCDE")
    years = list(range(2010, 2025))
    records = []
    site_N0 = {"A": 80, "B": 55, "C": 120, "D": 40, "E": 95}
    site_r  = {"A": 0.04, "B": 0.06, "C": 0.02, "D": 0.08, "E": 0.03}

    for s in sites:
        N = float(site_N0[s])
        for yr in years:
            sst      = round(RNG.normal(0, 0.8), 2)
            prey     = round(np.clip(RNG.normal(0.5 - 0.3 * sst, 0.2), 0, 1), 3)
            breeding = int(RNG.random() < 0.4)
            K        = 200
            r_eff    = site_r[s] * (1 + 0.5 * prey - 0.3 * abs(sst))
            N        = max(N + r_eff * N * (1 - N / K) + RNG.normal(0, 3), 5.0)
            effort   = int(RNG.integers(8, 25))
            p_cap    = np.clip(0.15 + 0.01 * effort - 0.05 * abs(sst), 0.05, 0.6)
            n_cap    = int(RNG.binomial(int(N), p_cap))
            n_marked = max(n_cap - RNG.integers(0, max(n_cap // 3, 1)), 1)
            p_recap  = np.clip(p_cap * 0.8, 0.02, 0.5)
            n_recap  = int(RNG.binomial(n_marked, p_recap))
            records.append({
                "site": s, "year": yr,
                "n_captured": n_cap, "n_recaptured": n_recap,
                "n_marked": n_marked, "effort": effort,
                "sst_anomaly": sst, "prey_index": prey,
                "breeding_flag": breeding, "true_N": round(N, 1),
            })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Data] 仮想データを保存: {csv_path}  ({len(df)} 行)")
    return df


# ─────────────────────────────────────────────
# 2. 特徴量エンジニアリング
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy().sort_values(["site", "year"]).reset_index(drop=True)
    sites_uniq = df["site"].unique()
    years_uniq = sorted(df["year"].unique())
    df["site_idx"] = df["site"].map({s: i for i, s in enumerate(sites_uniq)})
    df["year_idx"] = df["year"].map({y: i for i, y in enumerate(years_uniq)})
    for col in ["effort", "sst_anomaly", "prey_index"]:
        mu, sd = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sd
    return df, len(sites_uniq), len(years_uniq)


# ─────────────────────────────────────────────
# 3. 階層ベイズモデル
# ─────────────────────────────────────────────
def build_model(df: pd.DataFrame, n_sites: int, n_years: int) -> pm.Model:
    coords = {
        "site": [f"site_{i}" for i in range(n_sites)],
        "year": [f"yr_{i}"   for i in range(n_years)],
        "obs":  df.index.tolist(),
    }
    site_idx = df["site_idx"].values
    year_idx = df["year_idx"].values
    n_cap    = df["n_captured"].values
    n_marked = df["n_marked"].values
    n_recap  = df["n_recaptured"].values
    effort_z = df["effort_z"].values
    sst_z    = df["sst_anomaly_z"].values
    prey_z   = df["prey_index_z"].values

    with pm.Model(coords=coords) as model:

        # ── 事前分布: 個体数過程 ─────────────────────────────────
        log_N0     = pm.Normal("log_N0",    mu=4.5,  sigma=1.0,  dims="site")
        mu_r       = pm.Normal("mu_r",      mu=0.05, sigma=0.05)
        sigma_r    = pm.HalfNormal("sigma_r",    sigma=0.05)
        r_raw      = pm.Normal("r_raw",     mu=0,    sigma=1,    dims="site")
        r          = pm.Deterministic("r",  mu_r + r_raw * sigma_r, dims="site")
        sigma_proc = pm.HalfNormal("sigma_proc", sigma=0.2)

        # ── 状態空間: GaussianRandomWalk ────────────────────────
        # shape: (n_sites, n_years)
        log_N_rw = pm.GaussianRandomWalk(
            "log_N_rw",
            mu=r,
            sigma=sigma_proc,
            steps=n_years - 1,
            init_dist=pm.Normal.dist(mu=log_N0, sigma=0.01, shape=(n_sites,)),
        )

        # 2D AdvancedSubtensor を避けて 1D flatten + 1D インデックスで取得
        log_N_flat = log_N_rw.reshape((-1,))
        flat_idx   = site_idx * n_years + year_idx
        log_N_obs  = log_N_flat[flat_idx]
        N_obs = pm.Deterministic("N_obs", pm.math.exp(log_N_obs), dims="obs")

        # ── 事前分布: 捕獲確率 ──────────────────────────────────
        alpha       = pm.Normal("alpha",       mu=-1.5, sigma=1.0)
        beta_effort = pm.Normal("beta_effort", mu=0.3,  sigma=0.5)
        beta_sst    = pm.Normal("beta_sst",    mu=-0.2, sigma=0.5)
        beta_prey   = pm.Normal("beta_prey",   mu=0.4,  sigma=0.5)
        sigma_year  = pm.HalfNormal("sigma_year", sigma=0.3)
        eps_year    = pm.Normal("eps_year", mu=0, sigma=sigma_year, dims="year")
        kappa       = pm.Beta("kappa", alpha=5, beta=2)

        logit_p = (
            alpha
            + beta_effort * effort_z
            + beta_sst    * sst_z
            + beta_prey   * prey_z
            + eps_year[year_idx]
        )
        p_cap  = pm.Deterministic("p_cap",   pm.math.sigmoid(logit_p), dims="obs")
        p_recap= pm.Deterministic("p_recap",  p_cap * kappa,            dims="obs")

        # ── 尤度 ─────────────────────────────────────────────────
        lambda_cap = pm.Deterministic("lambda_cap", N_obs * p_cap, dims="obs")
        obs_cap    = pm.Poisson("obs_cap",   mu=lambda_cap, observed=n_cap,    dims="obs")
        obs_recap  = pm.Binomial("obs_recap", n=n_marked,   p=p_recap,
                                 observed=n_recap, dims="obs")

    return model


# ─────────────────────────────────────────────
# 4. サンプリング & 診断
# ─────────────────────────────────────────────
def run_sampling(model: pm.Model, draws: int = 1000, tune: int = 1000, chains: int = 2):
    with model:
        trace = pm.sample(
            draws=draws, tune=tune, chains=chains,
            cores=1,
            target_accept=0.9,
            return_inferencedata=True,
            random_seed=42,
        )
    return trace


def print_diagnostics(trace):
    print("\n── 収束診断 (R-hat & ESS) ──")
    summary = az.summary(
        trace,
        var_names=["mu_r", "sigma_r", "sigma_proc", "alpha",
                   "beta_effort", "beta_sst", "beta_prey", "kappa"],
        round_to=3,
    )
    print(summary.to_string())


# ─────────────────────────────────────────────
# 5. 可視化
# ─────────────────────────────────────────────
def plot_results(trace, df: pd.DataFrame, n_sites: int, n_years: int):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # ── suptitle ────────────────────────────────────────────────
    fig.suptitle(
        "ラッコ個体数推定モデル — 事後分布サマリ",
        fontsize=14, fontweight="bold",
        **_jp(14),
    )

    # ── (A) 地点別個体数推定 ────────────────────────────────────
    ax = axes[0, 0]
    N_obs_post = trace.posterior["N_obs"].values
    df_plot = df.copy()
    df_plot["N_mean"]   = N_obs_post.mean(axis=(0, 1))
    df_plot["N_hdi_lo"] = np.quantile(N_obs_post, 0.055, axis=(0, 1))
    df_plot["N_hdi_hi"] = np.quantile(N_obs_post, 0.945, axis=(0, 1))

    for s, grp in df_plot.groupby("site"):
        line, = ax.plot(grp["year"], grp["N_mean"], marker="o", lw=1.5)
        ax.fill_between(grp["year"], grp["N_hdi_lo"], grp["N_hdi_hi"], alpha=0.15)
        ax.plot(grp["year"], grp["true_N"], linestyle="--", alpha=0.4, color=line.get_color())

    ax.set_title("地点別 推定個体数（実線）vs 真値（破線）", **_jp(10))
    ax.set_xlabel("年",      **_jp(9))
    ax.set_ylabel("個体数",  **_jp(9))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    # 凡例を日本語フォントで作成
    legend_labels = [f"地点 {s}" for s in df_plot["site"].unique()]
    handles = ax.get_lines()[::3]  # 推定線のみ取得（3本ずつ: 推定・真値・fill）
    handles, _ = ax.get_legend_handles_labels()
    # 凡例を手動生成
    from matplotlib.lines import Line2D
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    legend_handles = [
        Line2D([0], [0], color=colors[i % len(colors)], lw=1.5, marker="o")
        for i, _ in enumerate(df_plot["site"].unique())
    ]
    leg = ax.legend(
        legend_handles,
        legend_labels,
        fontsize=7,
        prop=fm.FontProperties(fname=_FONT_PATH, size=7),
    )

    # ── (B) 地点別内的成長率 r ──────────────────────────────────
    ax = axes[0, 1]
    r_post = trace.posterior["r"].values
    r_flat = r_post.reshape(-1, n_sites)
    labels_jp = [f"地点 {s}" for s in list("ABCDE")]
    bp = ax.boxplot(r_flat, patch_artist=True,
                    boxprops=dict(facecolor="#5ba3c9", alpha=0.6))
    ax.set_xticks(range(1, n_sites + 1))
    ax.set_xticklabels(labels_jp,
                       fontproperties=fm.FontProperties(fname=_FONT_PATH, size=9))
    ax.axhline(0, color="red", linestyle="--", lw=0.8)
    ax.set_title("地点別 内的成長率 r の事後分布", **_jp(10))
    ax.set_ylabel("r", **_jp(9))

    # ── (C) 捕獲確率の事後分布 ──────────────────────────────────
    ax = axes[0, 2]
    p_cap_post = trace.posterior["p_cap"].values.reshape(-1, len(df))
    ax.hist(p_cap_post.mean(axis=0), bins=30, color="#e07b54",
            edgecolor="white", alpha=0.8)
    ax.set_title("捕獲確率 p_cap の事後平均分布", **_jp(10))
    ax.set_xlabel("p_cap", fontsize=9)
    ax.set_ylabel("観測数", **_jp(9))

    # ── (D) mu_r の事後分布 ─────────────────────────────────────
    ax = axes[1, 0]
    az.plot_posterior(trace, var_names=["mu_r"], ax=ax, color="#6ab187")
    ax.set_title("全体平均成長率 μ_r の事後分布", **_jp(10))
    # ArviZ が設定したテキストにもフォントを適用
    for txt in ax.texts:
        txt.set_fontproperties(JP_FONT)

    # ── (E) kappa の事後分布 ────────────────────────────────────
    ax = axes[1, 1]
    az.plot_posterior(trace, var_names=["kappa"], ax=ax, color="#b56db4")
    ax.set_title("再捕獲補正係数 κ の事後分布", **_jp(10))

    # ── (F) 全地点合計の個体数推移 ──────────────────────────────
    ax = axes[1, 2]
    total_true = df.groupby("year")["true_N"].sum()
    total_mean = df_plot.groupby("year")["N_mean"].sum()
    total_lo   = df_plot.groupby("year")["N_hdi_lo"].sum()
    total_hi   = df_plot.groupby("year")["N_hdi_hi"].sum()
    ax.plot(total_mean.index, total_mean.values, color="#2e86ab", lw=2,
            label="推定合計")
    ax.fill_between(total_lo.index, total_lo.values, total_hi.values,
                    alpha=0.2, color="#2e86ab")
    ax.plot(total_true.index, total_true.values, "k--", lw=1.2,
            label="真値合計")
    ax.set_title("全地点合計 個体数推移",  **_jp(10))
    ax.set_xlabel("年",          **_jp(9))
    ax.set_ylabel("個体数（合計）", **_jp(9))
    ax.legend(prop=fm.FontProperties(fname=_FONT_PATH, size=9))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    plt.tight_layout()
    out_path = "sea_otter_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n[Plot] 図を保存: {out_path}")
    plt.show()


# ─────────────────────────────────────────────
# 6. メイン
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  ラッコ個体数推定モデル (階層ベイズ 状態空間)")
    print("=" * 55)

    csv_path = "sea_otter_data.csv"
    if not Path(csv_path).exists():
        df = generate_virtual_data(csv_path)
    else:
        df = pd.read_csv(csv_path)
        print(f"[Data] 既存 CSV を読み込み: {csv_path}  ({len(df)} 行)")
    print(df.head(10).to_string(index=False))

    df, n_sites, n_years = preprocess(df)
    print(f"\n[Info] 地点数: {n_sites}, 年数: {n_years}, 観測数: {len(df)}")

    print("\n[Model] 階層ベイズモデルを構築中...")
    model = build_model(df, n_sites, n_years)
    print(pm.model_to_graphviz(model))

    print("\n[MCMC] サンプリング開始（tune=1000, draws=1000, chains=2）...")
    trace = run_sampling(model, draws=1000, tune=1000, chains=2)

    print_diagnostics(trace)
    plot_results(trace, df, n_sites, n_years)

    az.plot_trace(
        trace,
        var_names=["mu_r", "sigma_r", "sigma_proc", "alpha",
                   "beta_effort", "beta_sst", "beta_prey", "kappa"],
        compact=True,
    )
    plt.tight_layout()
    plt.savefig("sea_otter_trace.png", dpi=120)
    print("[Plot] トレースプロットを保存: sea_otter_trace.png")
    plt.show()


if __name__ == "__main__":
    main()