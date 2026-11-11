# -*- coding: utf-8 -*-
"""学習曲線と数値加工条件(DERIVATIVE / L2_NORMALIZED)のアンサンブル重みを可視化する。

【重要】元の学習コードは学習履歴を保存していないため、このファイルだけを実行しても
過去の学習曲線は復元できません。以下の「組み込み手順」を元コードに反映したうえで、
元の学習コードを再実行してください。

組み込み手順
============
1) 元コードの import 群の後に追加:
    from learning_diagnostics import save_training_history

2) run_single_target_experiment() 内で res = trainer.train_full_and_predict(...) の直後に追加:
    save_training_history(
        res["history"], config.output_root, target_man, eng_mode,
        trial + 1, unique_id
    )
   ※ res を del する前に保存します。

3) HybridEnsembleAggregator.run_ensemble() の「3. 重み付け平均による最終予測」内で、
   `weights_info = []` の直後に追加:
    weight_rows = []

   同じループ内で `weight = summary['mean_ccc'] / total_ccc_score` の直後に追加:
    weight_rows.append({
        "target_damage_man": os.path.basename(os.path.normpath(self.output_dir)),
        "eng_mode": eng_mode,
        "mean_validation_CCC": summary["mean_ccc"],
        "ensemble_weight": weight,
    })

   そして、そのループが終わった直後に追加:
    pd.DataFrame(weight_rows).to_csv(
        os.path.join(self.output_dir, "Condition_Weights.csv"),
        index=False, encoding="utf-8-sig"
    )

4) すべての対象損傷回数の学習が終わった後、ターミナルで実行:
    python learning_diagnostics.py --root ./result/distribution

出力
====
各対象損傷回数フォルダ内:
  TrainingHistory_<eng_mode>_trial<N>.csv
  LearningCurve_<eng_mode>.png
  LearningCurve_<eng_mode>.csv
  Condition_Weights.csv (元コードへの組み込み手順3で作成)
ルートフォルダ内:
  EnsembleWeights_by_Target.png
  EnsembleWeights_by_Target.csv

注意
====
・学習曲線は各試行の train/validation loss (MSE) と MAE を保存し、試行間の平均±標準偏差を描画。
・アンサンブル重みは元コードの仕様どおり、各数値加工条件の平均 validation CCC を正規化した値。
・重みは各エポックで更新される値ではなく、学習完了後に validation CCC から計算される値。
"""
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

ENG_MODES = ["DERIVATIVE", "L2_NORMALIZED"]
COLORS = {"DERIVATIVE": "#2563A6", "L2_NORMALIZED": "#C43C39"}
PLOT = {
    "figure_width": 9,
    "figure_height": 5.5,
    "title_fontsize": 14,
    "axis_label_fontsize": 16,
    "tick_fontsize": 10,
    "legend_fontsize": 16,
    "dpi": 200,
}

# 可視化の対象とする損傷回数の範囲 [万回] (両端を含む)
TARGET_RANGE_MAN = (400, 800)


def _in_target_range(value):
    """フォルダ名や値が TARGET_RANGE_MAN の範囲内か判定する。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return TARGET_RANGE_MAN[0] <= v <= TARGET_RANGE_MAN[1]

def save_training_history(history, output_root, target_damage_man, eng_mode, trial, unique_id=""):
    """Keras Historyを1試行分のCSVとして保存。元の学習ループから呼び出す。"""
    out_dir = Path(output_root) / str(target_damage_man)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = getattr(history, "history", {})
    if not hist:
        print("[診断ログ] 学習履歴が空のため保存をスキップしました。")
        return
    n_epochs = max(len(v) for v in hist.values())
    data = {"epoch": np.arange(1, n_epochs + 1)}
    for key, values in hist.items():
        data[key] = list(values) + [np.nan] * (n_epochs - len(values))
    df = pd.DataFrame(data)
    df["target_damage_man"] = target_damage_man
    df["eng_mode"] = eng_mode
    df["trial"] = trial
    df["unique_id"] = unique_id
    safe_mode = re.sub(r"[^A-Za-z0-9_-]", "_", str(eng_mode))
    path = out_dir / f"TrainingHistory_{safe_mode}_trial{trial}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[診断ログ] 学習履歴を保存: {path}")


def _font_config(ax):
    ax.tick_params(axis="both", labelsize=PLOT["tick_fontsize"])


def plot_learning_curves(target_dir):
    target_dir = Path(target_dir)
    files = list(target_dir.glob("TrainingHistory_*.csv"))
    if not files:
        return False
    all_df = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if {"epoch", "eng_mode"}.issubset(d.columns):
                all_df.append(d)
        except Exception as e:
            print(f"[警告] {f} を読めません: {e}")
    if not all_df:
        return False
    df = pd.concat(all_df, ignore_index=True)

    # MSE loss と MAE は単位・意味が異なるため、別々の図にする。
    for metric_pairs, metric_name, y_label in [
        ([ ("loss", "val_loss") ], "Loss", "MSE loss"),
        ([ ("mae", "val_mae") ], "MAE", "MAE (正規化ラベル空間)"),
    ]:
        fig, ax = plt.subplots(figsize=(PLOT["figure_width"], PLOT["figure_height"]))
        plotted = False
        for mode in ENG_MODES:
            mode_df = df[df["eng_mode"] == mode]
            if mode_df.empty:
                continue
            color = COLORS[mode]
            for key, label in metric_pairs:
                if key not in mode_df.columns:
                    continue
                summary = mode_df.groupby("epoch")[key].agg(["mean", "std"]).reset_index()
                x = summary["epoch"].to_numpy()
                y = summary["mean"].to_numpy()
                sd = summary["std"].fillna(0).to_numpy()
                # train / validation の線種を分け、色は加工条件を表す。
                is_val = key.startswith("val_")
                line_style = "--" if is_val else "-"
                label_text = f"{mode} - {'validation' if is_val else 'train'}"
                ax.plot(x, y, color=color, linestyle=line_style, linewidth=2, label=label_text)
                ax.fill_between(x, np.maximum(y - sd, 0), y + sd, color=color, alpha=0.12)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        target = target_dir.name
        ax.set_title(f"Learning Curve ({metric_name}) — target {target}万回", fontsize=PLOT["title_fontsize"])
        ax.set_xlabel("Epoch", fontsize=PLOT["axis_label_fontsize"])
        ax.set_ylabel(y_label, fontsize=PLOT["axis_label_fontsize"])
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=PLOT["legend_fontsize"], frameon=False)
        _font_config(ax)
        fig.tight_layout()
        out = target_dir / f"LearningCurve_{metric_name}.png"
        fig.savefig(out, dpi=PLOT["dpi"], bbox_inches="tight")
        plt.close(fig)
        print(f"[図] {out}")

    # 曲線の平均値もCSVに保存しておく（再作図・数値比較用）。
    numeric = [c for c in ["loss", "val_loss", "mae", "val_mae"] if c in df.columns]
    summary_rows = []
    for (mode, epoch), g in df.groupby(["eng_mode", "epoch"]):
        row = {"eng_mode": mode, "epoch": epoch, "n_trials": int(g["trial"].nunique()) if "trial" in g else len(g)}
        for col in numeric:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_std"] = g[col].std(ddof=1) if g[col].count() > 1 else 0.0
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(target_dir / "LearningCurve_summary.csv", index=False, encoding="utf-8-sig")
    return True


def plot_ensemble_weights(root_dir):
    root_dir = Path(root_dir)
    files = [
        f for f in root_dir.glob("*/Condition_Weights.csv")
        if _in_target_range(f.parent.name)
    ]
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
            required = {"target_damage_man", "eng_mode", "ensemble_weight"}
            if required.issubset(d.columns):
                frames.append(d)
        except Exception as e:
            print(f"[警告] {f} を読めません: {e}")
    if not frames:
        print("[注意] Condition_Weights.csv がありません。元コードへの組み込み手順3を適用し、実験を再実行してください。")
        return False

    df = pd.concat(frames, ignore_index=True)

    # ① 読み込んだ直後のデータを確認
    print("\n===== CSV読み込み直後 =====")
    print("eng_mode別の行数:")
    print(df["eng_mode"].value_counts(dropna=False))
    print("\n読み込み直後のデータ:")
    print(df[["target_damage_man", "eng_mode", "ensemble_weight"]].to_string(index=False))

    # ② eng_modeの余分な空白を除去
    df["eng_mode"] = df["eng_mode"].astype(str).str.strip()
    df["eng_mode"] = df["eng_mode"].replace({"L2": "L2_NORMALIZED"})

    # ③ 数値変換
    df["target_damage_man"] = pd.to_numeric(
        df["target_damage_man"], errors="coerce"
    )
    df["ensemble_weight"] = pd.to_numeric(
        df["ensemble_weight"], errors="coerce"
    )

    print("\n===== 数値変換後 =====")
    print(df.groupby("eng_mode", dropna=False).size())
    print("target_damage_man の欠損数:", df["target_damage_man"].isna().sum())
    print("ensemble_weight の欠損数:", df["ensemble_weight"].isna().sum())

    # ④ 欠損値除去後
    df = df.dropna(subset=["target_damage_man", "ensemble_weight"])

    # 200〜800万回のみ使用
    df = df[df["target_damage_man"].between(TARGET_RANGE_MAN[0], TARGET_RANGE_MAN[1])]

    print("\n===== 欠損値除去後 =====")
    print(df["eng_mode"].value_counts(dropna=False))

    # ⑤ 対象モードで絞り込み
    df = df[df["eng_mode"].isin(ENG_MODES)]
    print("\n===== 最終的な出力対象 =====")
    print(df["eng_mode"].value_counts(dropna=False))
    print(df[["target_damage_man", "eng_mode", "ensemble_weight"]].to_string(index=False))

    df = df.sort_values("target_damage_man")
    df.to_csv(root_dir / "EnsembleWeights_by_Target.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(PLOT["figure_width"], PLOT["figure_height"]))
    for mode in ENG_MODES:
        g = df[df["eng_mode"] == mode].sort_values("target_damage_man")
        if g.empty:
            continue
        ax.plot(g["target_damage_man"], g["ensemble_weight"], marker="o", linewidth=2.2,
                color=COLORS[mode], label=mode)
    ax.set_title("Ensemble Weight by Target Damage Count", fontsize=PLOT["title_fontsize"])
    ax.set_xlabel("Target damage count [万回]", fontsize=PLOT["axis_label_fontsize"])
    ax.set_ylabel("Ensemble weight (CCC-normalized)", fontsize=PLOT["axis_label_fontsize"])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=PLOT["legend_fontsize"], frameon=False)
    _font_config(ax)
    fig.tight_layout()
    out = root_dir / "EnsembleWeights_by_Target.png"
    fig.savefig(out, dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"[図] {out}")
    return True

def plot_prediction_boxplot(root_dir):
    """200〜800万回の各ターゲットについて、予測損傷回数の分布を箱ひげ図で描く。"""
    root_dir = Path(root_dir)
    data = {}
    for d in root_dir.iterdir():
        csv = d / "Final_Weighted_Ensemble_Result.csv"
        if d.is_dir() and _in_target_range(d.name) and csv.exists():
            pred = pd.to_numeric(pd.read_csv(csv)["Predicted_Damage"], errors="coerce").dropna()
            if not pred.empty:
                data[float(d.name)] = pred.values
    if not data:
        print("[注意] Final_Weighted_Ensemble_Result.csv がありません。")
        return False

    targets = sorted(data)
    # 損傷回数が多いほど濃い赤 (Redsの0.3〜0.9を使い、薄すぎる色を避ける)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(targets)))

    fig, ax = plt.subplots(figsize=(PLOT["figure_width"], PLOT["figure_height"]))
    bp = ax.boxplot([data[t] for t in targets], positions=range(len(targets)),
                    patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.8))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    # 正解値を各箱の位置に赤系の目印ではなく黒の破線で重ねる
    for i, t in enumerate(targets):
        ax.hlines(t * 10_000, i - 0.35, i + 0.35, colors="black", linestyles="--", linewidth=1.5,
                  label="Ground Truth" if i == 0 else None)

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([f"{int(t)}" for t in targets])
    ax.set_title("Predicted Damage Count Distribution by Target", fontsize=PLOT["title_fontsize"])
    ax.set_xlabel("Target damage count [万回]", fontsize=PLOT["axis_label_fontsize"])
    ax.set_ylabel("Predicted damage count [cycles]", fontsize=PLOT["axis_label_fontsize"])
    ax.set_ylim(bottom=0) # 上限も設定したいとき、ax.set_ylim(0, 9_000_000)のようにする
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:,.0f}"))

    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=PLOT["legend_fontsize"], frameon=False)
    _font_config(ax)
    fig.tight_layout()
    out = root_dir / "PredictionBoxplot_by_Target.png"
    fig.savefig(out, dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"[図] {out}")
    return True

def main():
    parser = argparse.ArgumentParser(description="学習曲線と数値加工条件の重みを可視化")
    parser.add_argument("--root", default="./result/distribution", help="実験結果のルートフォルダ")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"結果フォルダが見つかりません: {root.resolve()}")
    n_curves = 0
    target_dirs = [
        p for p in root.iterdir()
        if p.is_dir() and _in_target_range(p.name)
    ]
    for target_dir in sorted(target_dirs, key=lambda p: float(p.name)):
        if plot_learning_curves(target_dir):
            n_curves += 1
    weights_ok = plot_ensemble_weights(root)
    plot_prediction_boxplot(root)
    if n_curves == 0:
        print("[注意] 学習履歴CSVがありません。組み込み手順2を適用してから実験を再実行してください。")
    if weights_ok:
        print("診断図の出力が完了しました。")


if __name__ == "__main__":
    main()
