# このコードは測定用コードです。LCRメータから出力されたCSVファイルを基に、歯車の余寿命推定を行っていきます
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import glob
import os
import shutil
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import sys

import importlib.util
# ファイル数を減らすための動的インポート処理
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

# 警告を抑制
import warnings
warnings.simplefilter('ignore')

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
    'figure.autolayout': True
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

        # ★ LとRそれぞれの高相関周波数帯域を自動読み込み
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
# ./dataset/{コイル条件}/{訓練データ,テストデータ}/{csvファイル群}という構成になっています
# csvファイル群は12桁の命名規則で、例: L4000N1F0101.csv のようになっています
# 最初の2桁がコイル条件、次の3桁が歯車の損傷回数、次の2桁が歯車ID、次の3桁がコイルと接触する面、最後の2桁が試行回数を表しています
# 損傷回数は、以下のように変換する。000が0万回、040が400万回、080が800万回、800が1200万回、airが歯車無しの参照データ、damがpitching損傷の参照データとする
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
        self.train_air_avg = None  # 追加: trainのairデータをテストに流用するための変数

    def load_train_data(self):
        print("📥 Loading Training Data...")
        return self._load_data_generic(self.cfg.train_folders, is_blind=False)

    def load_blind_data(self):
        print("📥 Loading Blind Data...")
        return self._load_data_generic(self.cfg.blind_folders, is_blind=True)

    # DataManagerクラス内の _load_data_generic を置き換え、下に追加
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
                df_paired['filter_key'] = df_paired['folder'] + "-" + df_paired['GearID'] + "-" + df_paired[
                    'ToothID']
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
                label = 'Unknown'
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

            # 不要なStatus列などを捨て、必要なカラムだけを明示的に抽出する
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
        """
        CSVファイルのパスを収集

        Args:
            folders: フォルダリスト
            is_blind: Trueの場合はブラインドテストデータ

        Returns:
            (air_paths, data_paths) のタプル
        """
        air_paths = []
        data_paths = []

        for folder in folders:
            folder_path = os.path.join(self.cfg.base_folder, folder)
            if not os.path.exists(folder_path):
                print(f"⚠ Folder not found: {folder_path}")
                continue

            # train/test 問わず、指定フォルダ直下のサブフォルダを自動検出する
            print(f"   📂 Searching folders in: {folder_path}")
            target_paths = []
            try:
                # folder配下のすべてのサブフォルダを取得
                subfolders = [d for d in os.listdir(folder_path)
                              if os.path.isdir(os.path.join(folder_path, d)) and not d.startswith('.')]

                if subfolders:
                    target_paths = [os.path.join(folder_path, d) for d in sorted(subfolders)]
                    print(f"   ✓ Found {len(target_paths)} subdirectories: {', '.join(sorted(subfolders))}")
                else:
                    # サブフォルダがない場合は直下を探索パスにする
                    target_paths = [folder_path]
            except Exception as e:
                print(f"   ⚠ Error reading subfolders: {e}")
                target_paths = [folder_path]

            # 各ターゲットパスからファイルを収集
            for target_path in target_paths:
                if not os.path.exists(target_path):
                    continue

                # airファイルを探す
                air_files = glob.glob(os.path.join(target_path, "*air*.csv"))
                if air_files and is_blind:
                    print(f"      Air files in {os.path.basename(target_path)}: {len(air_files)}")
                air_paths.extend(air_files)

                # データファイルを探す
                for f in glob.glob(os.path.join(target_path, "*.csv")):
                    fname = os.path.basename(f)
                    if "air" in fname.lower():
                        continue
                    data_paths.append(f)

        return air_paths, data_paths

    def _create_air_average(self, paths):
        if not paths:
            return None

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

        if global_air_avg is None:
            print("   ⚠ Warning: No Air average data available for normalization!")
            return processed

        for p in paths:
            folder = os.path.basename(os.path.dirname(p))
            fname = os.path.splitext(os.path.basename(p))[0]
            try:
                df = pd.read_csv(p, encoding='utf-8-sig')
                # フォルダごとの照合を廃止し、グローバル基準を全データに適用
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
            y = np.zeros(len(labels))
        else:
            l_map = {'no_damaged': 0.0, 'damaged_040': 1 / 3, 'damaged_080': 2 / 3, 'damaged_800': 1.0}
            y = np.array([l_map.get(l, 0.0) for l in labels])
        return X, y, conditions, features

# ==============================================================================
# 4. Model Trainer Class (1D-CNN)
# ==============================================================================
class ModelTrainer:
    """1D-CNNモデルを使用した学習・予測クラス"""
    
    def __init__(self, config=None):
        self.config = config if config is not None else CNN1DConfig()
        self.model = CNN1D(self.config)
        
    def train_and_predict(self, X_train, y_train, X_blind):
        """
        モデルの学習と予測を実行
        
        Args:
            X_train: 学習データ (N, T, F)
            y_train: 学習ラベル (N,)
            X_blind: blindテストデータ (M, T, F)
        
        Returns:
            dict: 予測結果とメトリクス
        """
        result = self.model.train_and_predict(X_train, y_train, X_blind, return_val_metrics=True)
        
        return {
            "preds_inf": result['predictions'],
            "y_val_true": result['y_val_true'],
            "y_val_pred": result['y_val_pred'],
            "val_rmse": result['val_rmse'],
            "val_mae": result['val_mae'],
            "history": result['history']
        }

# ==============================================================================
# 6. Result Exporter Class
# ==============================================================================
class ResultExporter:
    """予測結果をCSVに出力するクラス"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
    def export_predictions(self, file_names, predictions):
        """
        予測結果をCSVに出力
        
        Args:
            file_names: ファイル名のリスト
            predictions: 予測値（損傷回数）の配列
        """
        df_result = pd.DataFrame({
            "FileName": file_names,
            "Predicted_Damage": predictions
        })
        
        # ファイル名から情報をパースしてソート
        try:
            df_result['GearID'] = df_result['FileName'].apply(lambda x: x.split('_')[1][5:7])
            df_result['ToothID'] = df_result['FileName'].apply(lambda x: x.split('_')[1][8:10])
            df_result = df_result.sort_values(by=['GearID', 'ToothID'])
            df_result = df_result.drop(columns=['GearID', 'ToothID'])
        except:
            pass  # パース失敗時はそのまま
        
        out_path = os.path.join(self.output_dir, "1D-CNN_Predictions.csv")
        df_result.to_csv(out_path, index=False)
        print(f"\n✅ Prediction results saved to: {out_path}")
        print(df_result.head())
        
        return out_path

# ==============================================================================
# 5. Data Statistics Analyzer Class
# ==============================================================================
class DataStatisticsAnalyzer:
    """trainデータとblindデータの統計情報を分析・表示"""
    
    def __init__(self, config):
        self.cfg = config
        
    def print_data_statistics(self, df_train, df_blind):
        """データの統計情報を表示"""
        print("\n" + "="*80)
        print("📊 DATA STATISTICS ANALYSIS")
        print("="*80)
        
        # Train データの統計
        print("\n【 TRAINING DATA 】")
        self._print_dataset_stats(df_train, "Train")
        
        # Blind データの統計
        print("\n【 BLIND DATA 】")
        self._print_dataset_stats(df_blind, "Blind")
        
        print("\n" + "="*80 + "\n")
    
    def _print_dataset_stats(self, df, dataset_name):
        """データセットの統計情報を表示"""
        if df is None or df.empty:
            print(f"⚠ {dataset_name} data is empty.")
            return
        
        # 総サンプル数
        total_samples = len(df['condition'].unique())
        print(f"\n📈 Total conditions: {total_samples}")
        
        # 損傷条件ごとのサンプル数
        print(f"\n🔍 Damage condition distribution:")
        label_counts = df.groupby('label')['condition'].nunique().sort_values(ascending=False)
        total_conditions = label_counts.sum()
        
        for label, count in label_counts.items():
            percentage = (count / total_conditions) * 100
            bar_length = int(percentage / 5)
            bar = "█" * bar_length
            print(f"  {label:20s}: {count:3d} [{percentage:5.1f}%] {bar}")
        
        # 歯車（GearID）ごとの条件数
        print(f"\n⚙️  Gear ID distribution:")
        gear_stats = df.groupby(['GearID', 'label'])['condition'].nunique().unstack(fill_value=0)
        print(gear_stats.to_string())
        
        # 周波数帯域での条件数
        if '[Hz]' in df.columns:
            freq_ranges = df.groupby('condition').apply(lambda x: (x['[Hz]'].min(), x['[Hz]'].max())).iloc[0]
            print(f"\n📡 Frequency range in data: {freq_ranges[0]/1e3:.1f} - {freq_ranges[1]/1e3:.1f} kHz")


# ==============================================================================
# 7. Main Execution Flow
# ==============================================================================
def main():
    print("🚀 Starting Model Training & Prediction...")
    try:
        print(f"\n{get_model_info()}\n")
    except Exception:
        pass

    config = ExperimentConfig()
    data_mgr = DataManager(config)
    stats_analyzer = DataStatisticsAnalyzer(config)

    # 1. Load Data
    df_train_full = data_mgr.load_train_data()
    df_blind_full = data_mgr.load_blind_data()

    if df_train_full is None or df_train_full.empty or df_blind_full is None or df_blind_full.empty:
        print("❌ Data is empty after processing. Check folder structures and air files.")
        return

    # 2. Print Data Statistics
    stats_analyzer.print_data_statistics(df_train_full, df_blind_full)

    # 3. Training & Prediction Loop (L & R)
    for target_name, params in config.target_features.items():
        print("\n" + "=" * 80)
        print(f"🎯 特性 {target_name} における予測評価 (周波数: {params['f_min'] / 1e3} - {params['f_max'] / 1e3} kHz)")
        print("=" * 80)

        # 周波数によるフィルタリング
        df_tr_curr = df_train_full[
            (df_train_full['[Hz]'] >= params['f_min']) & (df_train_full['[Hz]'] <= params['f_max'])].copy()
        df_bl_curr = df_blind_full[
            (df_blind_full['[Hz]'] >= params['f_min']) & (df_blind_full['[Hz]'] <= params['f_max'])].copy()

        # R面データの除外
        if config.drop_r_data:
            print("   ℹ Filtering out 'R' side data...")
            if 'side' in df_tr_curr.columns:
                df_tr_curr = df_tr_curr[df_tr_curr['side'] == 'F'].copy()
            if 'side' in df_bl_curr.columns:
                df_bl_curr = df_bl_curr[df_bl_curr['side'] == 'F'].copy()

        if df_tr_curr.empty or df_bl_curr.empty:
            print("   ⚠ Empty dataframe after filtering. Skipping.")
            continue

        # 前処理 & 特徴量生成
        preproc = StrictPreprocessor(config)

        # transformを削除し、直接渡す
        df_tr_fin = preproc.apply_feature_engineering(df_tr_curr, config.eng_mode)
        df_bl_fin = preproc.apply_feature_engineering(df_bl_curr, config.eng_mode)

        X_tr, y_tr, _, _ = preproc.generate_sequences(df_tr_fin, is_blind=False)
        X_bl, _, bl_conds, _ = preproc.generate_sequences(df_bl_fin, is_blind=True)

        if len(X_tr) == 0 or len(X_bl) == 0:
            print("   ⚠ Empty sequences. Skipping.")
            continue

        scale_factor = 1200
        preds_cnn_all = []
        preds_itr_all = []

        for trial in range(config.n_trials):
            print(f"   [Trial {trial + 1}/{config.n_trials}] 学習中...")
            # 1D-CNN
            cnn = CNN1D()
            res_cnn = cnn.train_and_predict(X_tr, y_tr, X_bl, return_val_metrics=False)
            preds_cnn_all.append(res_cnn['predictions'] * scale_factor)

            # i-Transformer
            itr = ITransformer()
            res_itr = itr.train_and_predict(X_tr, y_tr, X_bl)
            preds_itr_all.append(res_itr['predictions'] * scale_factor)

        # 結果の集計 (平均と標準偏差)
        cnn_mean = np.mean(preds_cnn_all, axis=0)
        cnn_std = np.std(preds_cnn_all, axis=0)

        itr_mean = np.mean(preds_itr_all, axis=0)
        itr_std = np.std(preds_itr_all, axis=0)

        # アンサンブル (5:5)
        ens_all = [(c + i) / 2 for c, i in zip(preds_cnn_all, preds_itr_all)]
        ens_mean = np.mean(ens_all, axis=0)
        ens_std = np.std(ens_all, axis=0)

        # 出力フォーマット
        models = [("1D-CNN", cnn_mean, cnn_std),
                  ("i-Transformer", itr_mean, itr_std),
                  ("Ensemble (5:5)", ens_mean, ens_std)]

        for model_name, mean_vals, std_vals in models:
            print(f"\n[{model_name}]")
            for idx, cond_name in enumerate(bl_conds):
                ground_truth = "Unknown"
                print(f"{cond_name}")
                print(f"予測結果：{mean_vals[idx]:.1f}万回±{std_vals[idx]:.1f}")
                print(f"正解：{ground_truth}\n")

    print("\n✅ All Finished.")


if __name__ == "__main__":
    main()