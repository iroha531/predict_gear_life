# このコードは測定用コードです。LCRメータから出力されたCSVファイルを基に、歯車の余寿命推定を行っていきます
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

import glob
import shutil
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import sys
from tqdm import tqdm

import importlib.util

# 警告・不要なログの完全抑制
import warnings

warnings.simplefilter('ignore')
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)  # ★ findfont警告を非表示


def load_model_module(module_name, file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cnn1d_module = load_model_module("cnn1d_module", "1D-CNN.py")
CNN1D = cnn1d_module.CNN1D
get_model_info = cnn1d_module.get_model_info

itrans_module = load_model_module("itrans_module", "i-transformer.py")
ITransformer = itrans_module.ITransformer

# ------------------------------------------------------------------------------
# ★ Plot Style Settings
# ------------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.5,
    'axes.edgecolor': 'black',
    'grid.linewidth': 1.0,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'lines.linewidth': 2.0,
    'figure.autolayout': True,
    'font.family': ['Noto Sans CJK JP', 'IPAexGothic', 'Hiragino Sans', 'Yu Gothic', 'sans-serif']
})


# ==============================================================================
# 1. 設定管理クラス
# ==============================================================================
class ExperimentConfig:
    def __init__(self):
        self.base_folder = "./dataset/L4"
        self.train_folders = ["train"]
        self.blind_folders = ["test"]
        self.manual_exclude_list = []

        self.eng_mode = "DERIVATIVE"
        self.drop_r_data = True

        config_path = "freq_config.json"
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                self.target_features = json.load(f)
            print(f"🔗 {config_path} から最適周波数領域を自動読み込みしました。")
        else:
            print(f"⚠ {config_path} が見つからないため、デフォルト周波数を設定します。")
            self.target_features = {
                'L': {'f_min': 10000, 'f_max': 50000},
                'R': {'f_min': 200000, 'f_max': 800000}
            }

        self.n_trials = 5
        self.patience = 15
        self.epochs = 50
        self.batch_size = 32
        self.output_dir = "./result/predict"


# ==============================================================================
# 2. データ管理クラス
# ==============================================================================
class DataManager:
    def __init__(self, config):
        self.cfg = config
        self.damage_map = {
            '000N1F': 'no_damaged', '000N1R': 'no_damaged',
            '000N2F': 'no_damaged', '000N2R': 'no_damaged',
            '000N3F': 'no_damaged', '000N3R': 'no_damaged',
            '040V3F': 'damaged_040', '000V3R': 'no_damaged',
            '080V2F': 'damaged_080', '000V2R': 'no_damaged',
            '800V1F': 'damaged_800', '000V1R': 'no_damaged',
            'damV1F': 'damaged_pitching', 'damV1R': 'no_damaged',
        }
        self.train_air_avg = None

    def load_train_data(self):
        return self._load_data_generic(self.cfg.train_folders, is_blind=False)

    def load_blind_data(self):
        return self._load_data_generic(self.cfg.blind_folders, is_blind=True)

    def _load_data_generic(self, folders, is_blind=False):
        air_paths, data_paths = self._collect_file_paths(folders, is_blind=is_blind)
        if not data_paths: return None

        air_avg = self._create_air_average(air_paths)
        if not is_blind:
            self.train_air_avg = air_avg
        elif is_blind and air_avg is None:
            air_avg = self.train_air_avg

        data_dicts = self._apply_air_normalization(data_paths, air_avg)
        df_paired = self._format_data(data_dicts, is_blind)

        if df_paired is not None and not df_paired.empty:
            if not is_blind:
                df_paired = df_paired[df_paired['label'] != 'damaged_pitching'].copy()
            if self.cfg.manual_exclude_list:
                df_paired['filter_key'] = df_paired['folder'] + "-" + df_paired['GearID'] + "-" + df_paired['ToothID']
                df_paired = df_paired[~df_paired['filter_key'].isin(self.cfg.manual_exclude_list)].copy()
            df_paired['group_key'] = df_paired['folder'] + "_" + df_paired['GearID'] + "_" + df_paired['ToothID']

        return df_paired

    def _format_data(self, data_dicts, is_blind):
        target_cols = ['LS[H]_airdiff', '|Z|[ohm]_airdiff', 'RS[ohm]_airdiff', 'Q(s)[-]_airdiff', 'X[ohm]_airdiff',
                       'PhaseZ[deg]_airdiff']
        all_data = []
        for (folder, fname), df in data_dicts.items():
            if len(fname) < 10: continue

            cond_code, gear_id, side, tooth_id = fname[2:5], fname[5:7], fname[7], fname[8:10]

            if is_blind:
                if "2_million" in folder:
                    label = 200.0
                elif "6_million" in folder:
                    label = 600.0
                else:
                    label = 0.0
            else:
                if cond_code == "000":
                    label = "no_damaged"
                elif cond_code == "040":
                    label = "damaged_040"
                elif cond_code == "080":
                    label = "damaged_080"
                elif cond_code == "800":
                    label = "damaged_800"
                else:
                    label = self.damage_map.get(cond_code + gear_id + side, 'no_label')

            available_targets = [c for c in target_cols if c in df.columns]
            keep_cols = ['[Hz]', 'folder'] + available_targets

            df_out = df[keep_cols].copy()
            df_out['condition'] = fname
            df_out['side'] = side
            df_out['label'] = label
            df_out['GearID'] = gear_id
            df_out['ToothID'] = tooth_id
            all_data.append(df_out)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def _collect_file_paths(self, folders, is_blind=False):
        air_paths = []
        data_paths = []
        for folder in folders:
            folder_path = os.path.join(self.cfg.base_folder, folder)
            if not os.path.exists(folder_path): continue

            target_paths = []
            try:
                subfolders = [d for d in os.listdir(folder_path) if
                              os.path.isdir(os.path.join(folder_path, d)) and not d.startswith('.')]
                if subfolders:
                    target_paths = [os.path.join(folder_path, d) for d in sorted(subfolders)]
                else:
                    target_paths = [folder_path]
            except Exception:
                target_paths = [folder_path]

            for target_path in target_paths:
                if not os.path.exists(target_path): continue
                air_files = glob.glob(os.path.join(target_path, "*air*.csv"))
                air_paths.extend(air_files)
                for f in glob.glob(os.path.join(target_path, "*.csv")):
                    fname = os.path.basename(f)
                    if "air" in fname.lower(): continue
                    data_paths.append(f)

        return air_paths, data_paths

    def _create_air_average(self, paths):
        if not paths: return None
        dfs = []
        cols = ['[Hz]', 'LS[H]', '|Z|[ohm]', 'RS[ohm]', 'Q(s)[-]', 'X[ohm]', 'PhaseZ[deg]']
        for p in paths:
            try:
                df = pd.read_csv(p, encoding='utf-8-sig')[cols].dropna()
                dfs.append(df)
            except:
                continue

        if not dfs: return None
        combined = pd.concat(dfs)
        avg = combined.groupby('[Hz]', as_index=False).mean()
        avg.rename(columns={'LS[H]': 'L0', '|Z|[ohm]': 'Z0', 'RS[ohm]': 'R0', 'Q(s)[-]': 'Q0', 'X[ohm]': 'X0',
                            'PhaseZ[deg]': 'PhaseZ0'}, inplace=True)
        return avg

    def _apply_air_normalization(self, paths, global_air_avg):
        processed = {}
        mapping = {'LS[H]': 'L0', '|Z|[ohm]': 'Z0', 'RS[ohm]': 'R0', 'Q(s)[-]': 'Q0', 'X[ohm]': 'X0',
                   'PhaseZ[deg]': 'PhaseZ0'}

        if global_air_avg is None: return processed
        for p in paths:
            folder = os.path.basename(os.path.dirname(p))
            fname = os.path.splitext(os.path.basename(p))[0]
            try:
                df = pd.read_csv(p, encoding='utf-8-sig')
                df = pd.merge(df, global_air_avg, on='[Hz]', how='left')
                for col, air_col in mapping.items():
                    if col in df.columns and air_col in df.columns:
                        df[f"{col}_airdiff"] = df[col] - df[air_col]
                df['folder'] = folder
                df['original_filename'] = fname
                processed[(folder, fname)] = df
            except:
                continue
        return processed


# ==============================================================================
# 3. 前処理クラス
# ==============================================================================
class StrictPreprocessor:
    def __init__(self, config):
        self.cfg = config

    def apply_feature_engineering(self, df_input, eng_mode):
        exclude = ["condition", "label", "folder", "[Hz]", "side", "GearID", "ToothID", "group_key", "is_rescued",
                   "map_key"]
        feature_cols = [c for c in df_input.columns if c not in exclude]

        df_sorted = df_input.sort_values(by=['condition', '[Hz]'])
        processed_dfs = []

        for name, group in df_sorted.groupby('condition'):
            g = group.copy()
            g_res = g[['condition', 'label', 'folder', '[Hz]']].copy()
            for col in feature_cols:
                vals = g[col].values
                if eng_mode == "DERIVATIVE":
                    g_res[col] = np.gradient(vals, g['[Hz]'].values)
                elif eng_mode == "L2_NORMALIZED":
                    norm = np.sqrt(np.sum(vals ** 2))
                    g_res[col] = vals / norm if norm > 0 else vals
                else:
                    g_res[col] = vals
            processed_dfs.append(g_res)
        return pd.concat(processed_dfs)

    def generate_sequences(self, df, is_blind=False):
        sequences, labels, conditions = [], [], []
        exclude = ["condition", "label", "folder", "[Hz]", "side", "GearID", "ToothID", "group_key", "is_rescued",
                   "map_key"]
        features = [c for c in df.columns if c not in exclude]

        for name, group in df.groupby('condition'):
            if group.empty: continue
            sequences.append(group[features].values)
            labels.append(group['label'].iloc[0])
            conditions.append(name)

        X = np.array(sequences)
        if is_blind:
            y = np.array([float(l) for l in labels])
        else:
            l_map = {'no_damaged': 0.0, 'damaged_040': 1 / 3, 'damaged_080': 2 / 3, 'damaged_800': 1.0}
            y = np.array([l_map.get(l, 0.0) for l in labels])
        return X, y, conditions, features


# ==============================================================================
# 4. Result Exporter Class
# ==============================================================================
class ResultExporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def export_histogram(self, models_results, target_name, true_val):
        plt.figure(figsize=(10, 6))
        colors = ['steelblue', 'tomato', 'forestgreen']
        for (model_name, mean_vals, _), color in zip(models_results, colors):
            sns.histplot(mean_vals, kde=True, label=model_name, color=color, alpha=0.5, bins=8)

        plt.axvline(x=true_val, color='red', linestyle='--', linewidth=2.5, label=f'True Value ({true_val}万回)')
        plt.title(f"Prediction Distribution ({true_val}万回テストデータ / 特性: {target_name})", fontsize=14)
        plt.xlabel("予測損傷回数 [万回]", fontsize=12)
        plt.ylabel("頻度", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        out_path = os.path.join(self.output_dir, f"Hist_{target_name}_{int(true_val)}M.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

    def export_ensemble_process(self, cond_names, cnn_m, cnn_s, itr_m, itr_s, ens_m, ens_s, true_val, target_name):
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(cond_names))

        plt.errorbar(x_pos - 0.15, cnn_m, yerr=cnn_s, fmt='o', color='steelblue', label='1D-CNN', alpha=0.7, capsize=3)
        plt.errorbar(x_pos + 0.15, itr_m, yerr=itr_s, fmt='^', color='tomato', label='i-Transformer', alpha=0.7,
                     capsize=3)
        plt.scatter(x_pos, ens_m, color='gold', marker='*', s=150, edgecolor='black', label='Weighted Ensemble',
                    zorder=5)
        plt.axhline(y=true_val, color='red', linestyle='-', linewidth=2, label=f'True Value ({true_val})', zorder=1)

        plt.title(f"Ensemble Process by Inverse Variance ({true_val}万回データ / 特性: {target_name})", fontsize=14)
        plt.xlabel("Test Sample Index", fontsize=12)
        plt.ylabel("予測損傷回数 [万回]", fontsize=12)

        # アンダースコア分割をやめ、文字のインデックスから直接「V4-01」のようなラベルを生成
        labels = [f"{c[5:7]}-{c[8:10]}" if len(c) >= 10 else c for c in cond_names]
        plt.xticks(x_pos, labels, rotation=45, fontsize=9)

        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        out_path = os.path.join(self.output_dir, f"Ensemble_Process_{target_name}_{int(true_val)}M.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


# ==============================================================================
# 5. Main Execution Flow
# ==============================================================================
def main():
    print("🚀 Starting Model Training & Prediction...\n")

    config = ExperimentConfig()
    data_mgr = DataManager(config)
    exporter = ResultExporter(config.output_dir)

    print("📥 Loading Data...")
    df_train_full = data_mgr.load_train_data()
    df_blind_full = data_mgr.load_blind_data()

    if df_train_full is None or df_train_full.empty or df_blind_full is None or df_blind_full.empty:
        print("❌ Data is empty after processing.")
        return

    # 3. Training & Prediction Loop (L & R)
    for target_name, params in config.target_features.items():
        print("\n" + "=" * 80)
        print(f"🎯 特性 {target_name} における予測評価 (周波数: {params['f_min'] / 1e3} - {params['f_max'] / 1e3} kHz)")
        print("=" * 80)

        df_tr_curr = df_train_full[
            (df_train_full['[Hz]'] >= params['f_min']) & (df_train_full['[Hz]'] <= params['f_max'])].copy()
        df_bl_curr = df_blind_full[
            (df_blind_full['[Hz]'] >= params['f_min']) & (df_blind_full['[Hz]'] <= params['f_max'])].copy()

        if config.drop_r_data:
            if 'side' in df_tr_curr.columns:
                df_tr_curr = df_tr_curr[df_tr_curr['side'] == 'F'].copy()
            if 'side' in df_bl_curr.columns:
                df_bl_curr = df_bl_curr[df_bl_curr['side'] == 'F'].copy()

        if df_tr_curr.empty or df_bl_curr.empty:
            continue

        preproc = StrictPreprocessor(config)
        df_tr_fin = preproc.apply_feature_engineering(df_tr_curr, config.eng_mode)
        df_bl_fin = preproc.apply_feature_engineering(df_bl_curr, config.eng_mode)

        X_tr, y_tr, _, _ = preproc.generate_sequences(df_tr_fin, is_blind=False)
        X_bl, y_bl, bl_conds, _ = preproc.generate_sequences(df_bl_fin, is_blind=True)

        if len(X_tr) == 0 or len(X_bl) == 0:
            continue

        scale_factor = 1200
        preds_cnn_all = []
        preds_itr_all = []

        # ★ 修正: タスクを2倍に細分化し、リアルタイムでステータスを表示
        total_steps = config.n_trials * 2
        with tqdm(total=total_steps, desc=f"   ⚙️ モデル学習中", leave=False) as pbar:
            for trial in range(config.n_trials):
                # 1D-CNNの学習
                pbar.set_postfix({"Trial": f"{trial + 1}/{config.n_trials}", "Model": "1D-CNN"})
                cnn = CNN1D()
                res_cnn = cnn.train_and_predict(X_tr, y_tr, X_bl, return_val_metrics=False)
                preds_cnn_all.append(res_cnn['predictions'] * scale_factor)
                pbar.update(1)

                # i-Transformerの学習
                pbar.set_postfix({"Trial": f"{trial + 1}/{config.n_trials}", "Model": "i-Trans"})
                itr = ITransformer()
                res_itr = itr.train_and_predict(X_tr, y_tr, X_bl)
                preds_itr_all.append(res_itr['predictions'] * scale_factor)
                pbar.update(1)

        # 結果の集計（全体）
        cnn_mean = np.mean(preds_cnn_all, axis=0)
        cnn_std = np.std(preds_cnn_all, axis=0)
        itr_mean = np.mean(preds_itr_all, axis=0)
        itr_std = np.std(preds_itr_all, axis=0)

        # 逆分散加重平均
        eps = 1e-6
        w_cnn = 1.0 / (cnn_std ** 2 + eps)
        w_itr = 1.0 / (itr_std ** 2 + eps)

        ens_mean = (w_cnn * cnn_mean + w_itr * itr_mean) / (w_cnn + w_itr)
        ens_std = np.sqrt(1.0 / (w_cnn + w_itr))

        # ★ 修正: 正解ラベルごとに結果を分割し、平均・中央値・標準偏差のみを出力する
        unique_truths = np.unique(y_bl)
        for true_val in unique_truths:
            idx = (y_bl == true_val)
            if not np.any(idx): continue

            print(f"\n【 {true_val}万回 テストデータ予測結果 】")
            models = [
                ("1D-CNN", cnn_mean[idx], cnn_std[idx]),
                ("i-Transformer", itr_mean[idx], itr_std[idx]),
                ("Ensemble (Inv-Var)", ens_mean[idx], ens_std[idx])
            ]

            for model_name, m_vals, s_vals in models:
                avg_val = np.mean(m_vals)
                med_val = np.median(m_vals)
                std_val = np.std(m_vals)
                print(
                    f"  [{model_name:<18}] 予測分布: 平均 {avg_val:>5.1f}万回 | 中央値 {med_val:>5.1f}万回 | 標準偏差 {std_val:>4.1f}")

            # 画像の保存
            exporter.export_histogram(models, target_name, true_val)
            conds_idx = np.array(bl_conds)[idx]
            exporter.export_ensemble_process(
                conds_idx, cnn_mean[idx], cnn_std[idx], itr_mean[idx], itr_std[idx],
                ens_mean[idx], ens_std[idx], true_val, target_name
            )

    print("\n✅ All Finished.")


if __name__ == "__main__":
    main()