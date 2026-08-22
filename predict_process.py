# このコードは測定用コードです。LCRメータから出力されたCSVファイルを基に、歯車の余寿命推定を行っていきます
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

        # ★追加: 生信号可視化の代表サンプル数
        self.n_raw_samples_per_label = 3

        # ★追加: 前処理パイプラインの有効/無効切り替え
        self.enable_phase_rotation = True
        self.enable_filtering = True
        self.enable_normalization = True

        # ★追加: フィルタリング設定
        self.filter_window = 5      # 平滑化の移動窓幅（奇数推奨）
        self.detrend_degree = 1     # デトレンド（ドリフト除去）の多項式次数

        # ★追加: 位相回転・フィルタ・正規化の対象とする複素インピーダンス列
        self.impedance_r_col = 'RS[ohm]_airdiff'
        self.impedance_x_col = 'X[ohm]_airdiff'

        # ★追加: 前処理段階の可視化に使う対象列
        self.stage_plot_cols = (
            'RS[ohm]_airdiff', 'X[ohm]_airdiff',
            '|Z|[ohm]_airdiff', 'PhaseZ[deg]_airdiff',
        )


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
# 2.5 ★追加: 前処理段階ごとの信号可視化クラス
# ==============================================================================
class StageSignalVisualizer:
    """
    前処理の各段階（生データ → 位相回転 → フィルタリング → 正規化 → 特徴量エンジニアリング）
    について、損傷レベルごとに代表サンプルを抽出し、信号の変化を画像として保存するクラス。

    出力ディレクトリは段階ごとに分けて保存する:
        <output_dir>/preprocessing_stages/01_raw/...
        <output_dir>/preprocessing_stages/02_phase_rotation/...
        <output_dir>/preprocessing_stages/03_filtering/...
        <output_dir>/preprocessing_stages/04_normalization/...
        <output_dir>/preprocessing_stages/05_feature_engineering/<target_name>/...
    """

    def __init__(self, base_output_dir):
        self.base_dir = os.path.join(base_output_dir, "preprocessing_stages")
        os.makedirs(self.base_dir, exist_ok=True)

    def plot_stage(self, df, dataset_name, stage_dir_name, value_cols, n_samples=3,
                   x_col='[Hz]', xscale='log', stage_title=""):
        """
        1つの前処理段階について、損傷レベルごとに代表サンプルをプロットする。

        Parameters
        ----------
        df : pd.DataFrame
            対象データ（'condition', 'label', x_col, value_cols を含む）
        dataset_name : str
            "train" / "test" などの識別子（ファイル名に使用）
        stage_dir_name : str
            段階名（例: "01_raw"）。この名前でサブディレクトリを作成する。
        value_cols : list or tuple
            プロット対象の列名
        n_samples : int
            各損傷レベルから抽出する代表サンプル数
        stage_title : str
            グラフタイトルに表示する段階名（日本語表記）
        """
        if df is None or df.empty:
            print(f"⚠ {dataset_name}/{stage_dir_name}: データが空のためプロットをスキップします。")
            return

        out_dir = os.path.join(self.base_dir, stage_dir_name)
        os.makedirs(out_dir, exist_ok=True)

        try:
            labels = sorted(df['label'].unique(), key=lambda v: (isinstance(v, str), v))
        except TypeError:
            labels = list(df['label'].unique())

        for label in labels:
            sub = df[df['label'] == label]
            conditions = sub['condition'].unique()
            if len(conditions) == 0:
                continue
            picked = conditions[:n_samples]

            for col in value_cols:
                if col not in df.columns:
                    continue

                plt.figure(figsize=(9, 5))
                for cond in picked:
                    cond_df = sub[sub['condition'] == cond].sort_values(x_col)
                    if cond_df.empty:
                        continue
                    plt.plot(cond_df[x_col], cond_df[col], marker='o', markersize=3,
                             linewidth=1.5, label=cond)

                if xscale == 'log':
                    plt.xscale('log')
                plt.xlabel(x_col)
                plt.ylabel(col)
                plt.title(f"[{dataset_name}] {stage_title}\nlabel={label} (代表{len(picked)}サンプル)")
                plt.legend(fontsize=8, loc='best')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                safe_label = str(label).replace('.', '_')
                safe_col = (col.replace('|', '').replace('[', '')
                            .replace(']', '').replace('/', '_').replace('(', '').replace(')', ''))
                out_path = os.path.join(out_dir, f"{dataset_name}_{safe_label}_{safe_col}.png")
                plt.savefig(out_path, dpi=150)
                plt.close()

        print(f"   💾 [{stage_dir_name}] {dataset_name} の段階別プロットを保存: {out_dir}")


# ==============================================================================
# 2.6 ★追加: 前処理パイプラインクラス
# ==============================================================================
class PreprocessingPipeline:
    """
    生データ（大気補正差分まで適用済み）からモデル入力までの前処理を担うクラス。

    ステップ:
        1. 位相回転 (Phase Rotation)
            R-X平面（実部・虚部）を、健全材(no_damaged)の平均ベクトル方向を基準に回転させ、
            リフトオフ変動方向（PR_liftoff）と、それに直交する欠陥感度の高い方向（PR_defect）
            に信号を分離する。回転角 theta は必ず「学習データの健全材」からのみ算出し、
            train / test 双方に同一の回転を適用することでリークを防ぐ。

        2. フィルタリング (Filtering)
            条件(condition)ごとに周波数方向で移動平均による平滑化を行い、
            電源ノイズ等に由来する高周波ランダムノイズを除去する。
            続いて多項式によるデトレンドを行い、温度変化等に由来する
            緩やかな低周波ドリフトを除去する。

        3. 正規化 (Normalization)
            学習データの平均・標準偏差を用いて各特徴量をZスコア標準化する。
            スケーラーは学習データにのみfitし、testには学習データの統計量を適用する。
    """

    def __init__(self, config):
        self.cfg = config
        self.theta_ = 0.0
        self.scaler_mean_ = {}
        self.scaler_std_ = {}

    # --------------------------------------------------------------------
    # (i) 位相回転
    # --------------------------------------------------------------------
    def fit_phase_rotation(self, df_train):
        col_r, col_x = self.cfg.impedance_r_col, self.cfg.impedance_x_col
        if col_r not in df_train.columns or col_x not in df_train.columns:
            self.theta_ = 0.0
            return

        ref = df_train[df_train['label'] == 'no_damaged']
        if ref.empty:
            ref = df_train  # 健全材ラベルが無い場合は全データで代用

        mean_r = ref[col_r].mean()
        mean_x = ref[col_x].mean()
        # 健全材の平均ベクトル方向を「リフトオフ変動が支配的な方向」とみなし、回転角を決定
        self.theta_ = float(np.arctan2(mean_x, mean_r))
        print(f"   🔄 位相回転角 theta = {np.degrees(self.theta_):.2f} deg （学習データの健全材から算出）")

    def apply_phase_rotation(self, df):
        col_r, col_x = self.cfg.impedance_r_col, self.cfg.impedance_x_col
        df = df.copy()
        if col_r not in df.columns or col_x not in df.columns:
            return df

        cos_t, sin_t = np.cos(self.theta_), np.sin(self.theta_)
        r = df[col_r].values
        x = df[col_x].values
        # PR_liftoff: リフトオフ変動が集約される軸
        # PR_defect : リフトオフ方向と直交し、欠陥信号が際立つ軸
        df['PR_liftoff'] = r * cos_t + x * sin_t
        df['PR_defect'] = -r * sin_t + x * cos_t
        return df

    # --------------------------------------------------------------------
    # (ii) フィルタリング（平滑化 + デトレンド）
    # --------------------------------------------------------------------
    def apply_filtering(self, df, feature_cols):
        df = df.copy()
        window = self.cfg.filter_window
        degree = self.cfg.detrend_degree

        for name, group in df.groupby('condition'):
            g_sorted = group.sort_values('[Hz]')
            idx = g_sorted.index
            n = len(idx)
            if n < 3:
                continue
            eff_window = min(window, n if n % 2 == 1 else n - 1)
            if eff_window < 3:
                eff_window = 3 if n >= 3 else n

            freqs = g_sorted['[Hz]'].values

            for col in feature_cols:
                if col not in df.columns:
                    continue
                vals = g_sorted[col].values

                # --- 高周波ノイズ除去: 移動平均による平滑化 ---
                smoothed = pd.Series(vals).rolling(
                    window=eff_window, center=True, min_periods=1).mean().values

                # --- 低周波ドリフト除去: 多項式トレンドを差し引くデトレンド ---
                try:
                    eff_degree = min(degree, max(0, n - 1))
                    trend_coef = np.polyfit(freqs, smoothed, eff_degree)
                    trend = np.polyval(trend_coef, freqs)
                    detrended = smoothed - trend
                except Exception:
                    detrended = smoothed

                df.loc[idx, col] = detrended

        return df

    # --------------------------------------------------------------------
    # (iii) 正規化（Zスコア標準化）
    # --------------------------------------------------------------------
    def fit_normalization(self, df_train, feature_cols):
        for col in feature_cols:
            if col in df_train.columns:
                mean_v = df_train[col].mean()
                std_v = df_train[col].std()
                self.scaler_mean_[col] = mean_v
                self.scaler_std_[col] = std_v if std_v and std_v > 1e-12 else 1.0

    def apply_normalization(self, df, feature_cols):
        df = df.copy()
        for col in feature_cols:
            if col in df.columns and col in self.scaler_mean_:
                df[col] = (df[col] - self.scaler_mean_[col]) / self.scaler_std_[col]
        return df

    # --------------------------------------------------------------------
    # 一括実行ヘルパー
    # --------------------------------------------------------------------
    def get_feature_cols(self, df):
        exclude = ["condition", "label", "folder", "[Hz]", "side", "GearID", "ToothID",
                   "group_key", "is_rescued", "map_key", "original_filename"]
        return [c for c in df.columns if c not in exclude]

    def run(self, df_train_raw, df_blind_raw, visualizer: "StageSignalVisualizer" = None):
        """
        生データ(df_train_raw, df_blind_raw)から、位相回転→フィルタリング→正規化 の
        全段階を順に適用し、各段階の結果を返す。visualizer が渡された場合は各段階の
        代表サンプルを画像として保存する。
        """
        stage_cols = list(self.cfg.stage_plot_cols)
        n_samples = self.cfg.n_raw_samples_per_label

        # --- Stage 01: 生データ（大気補正差分まで） ---
        if visualizer is not None:
            visualizer.plot_stage(df_train_raw, "train", "01_raw", stage_cols,
                                  n_samples=n_samples, stage_title="① 生データ（大気補正差分）")
            visualizer.plot_stage(df_blind_raw, "test", "01_raw", stage_cols,
                                  n_samples=n_samples, stage_title="① 生データ（大気補正差分）")

        df_train_cur = df_train_raw
        df_blind_cur = df_blind_raw

        # --- Stage 02: 位相回転 ---
        if self.cfg.enable_phase_rotation:
            self.fit_phase_rotation(df_train_cur)
            df_train_cur = self.apply_phase_rotation(df_train_cur)
            df_blind_cur = self.apply_phase_rotation(df_blind_cur)

            pr_cols = [c for c in ['PR_liftoff', 'PR_defect'] if c in df_train_cur.columns]
            plot_cols_stage2 = stage_cols + pr_cols
            if visualizer is not None:
                visualizer.plot_stage(df_train_cur, "train", "02_phase_rotation", plot_cols_stage2,
                                      n_samples=n_samples, stage_title="② 位相回転（リフトオフ方向の分離）")
                visualizer.plot_stage(df_blind_cur, "test", "02_phase_rotation", plot_cols_stage2,
                                      n_samples=n_samples, stage_title="② 位相回転（リフトオフ方向の分離）")

        # --- Stage 03: フィルタリング ---
        feature_cols = self.get_feature_cols(df_train_cur)
        if self.cfg.enable_filtering:
            df_train_cur = self.apply_filtering(df_train_cur, feature_cols)
            df_blind_cur = self.apply_filtering(df_blind_cur, feature_cols)

            if visualizer is not None:
                visualizer.plot_stage(df_train_cur, "train", "03_filtering", feature_cols,
                                      n_samples=n_samples, stage_title="③ フィルタリング（平滑化+デトレンド）")
                visualizer.plot_stage(df_blind_cur, "test", "03_filtering", feature_cols,
                                      n_samples=n_samples, stage_title="③ フィルタリング（平滑化+デトレンド）")

        # --- Stage 04: 正規化 ---
        if self.cfg.enable_normalization:
            self.fit_normalization(df_train_cur, feature_cols)
            df_train_cur = self.apply_normalization(df_train_cur, feature_cols)
            df_blind_cur = self.apply_normalization(df_blind_cur, feature_cols)

            if visualizer is not None:
                visualizer.plot_stage(df_train_cur, "train", "04_normalization", feature_cols,
                                      n_samples=n_samples, stage_title="④ 正規化（Zスコア標準化）")
                visualizer.plot_stage(df_blind_cur, "test", "04_normalization", feature_cols,
                                      n_samples=n_samples, stage_title="④ 正規化（Zスコア標準化）")

        return df_train_cur, df_blind_cur


# ==============================================================================
# 2.7 ★追加: 前処理フローチャート生成
# ==============================================================================
def draw_preprocessing_flowchart(output_dir):
    """
    生データからモデル入力に至るまでの前処理パイプラインをフローチャート画像として出力する。
    """
    stage_dir = os.path.join(output_dir, "preprocessing_stages", "00_flowchart")
    os.makedirs(stage_dir, exist_ok=True)

    stages = [
        ("① LCRメータ生データ\n(CSV: L, R, |Z|, Q, X, PhaseZ)",
         "各歯車・各歯番号ごとに\n周波数スイープ計測"),
        ("② 大気補正差分\n(Air Normalization)",
         "健全基準(air)との差分を取り、\n個体差・絶対値ズレを補正"),
        ("③ 位相回転\n(Phase Rotation)",
         "R-X平面を回転し、\nリフトオフ方向と欠陥方向を分離"),
        ("④ フィルタリング\n(Smoothing + Detrend)",
         "高周波ノイズを平滑化、\n低周波ドリフトを除去"),
        ("⑤ 正規化\n(Standardization)",
         "学習データの統計量で\nZスコア標準化"),
        ("⑥ 特徴量エンジニアリング\n(周波数微分 等)",
         "局所的な信号変化を\n強調する特徴量に変換"),
        ("⑦ モデル入力\n(X_tr / X_bl)",
         "1D-CNN, i-Transformer へ入力"),
    ]

    n = len(stages)
    fig, ax = plt.subplots(figsize=(9, 2.2 * n))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n * 2 + 1)
    ax.axis('off')

    box_w, box_h = 5.4, 1.3
    x_center = 3.2

    y_positions = []
    for i, (title, desc) in enumerate(stages):
        y = (n - i) * 2
        y_positions.append(y)
        box = mpatches.FancyBboxPatch(
            (x_center - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor='#DCEBFA', edgecolor='black', linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x_center, y, title, ha='center', va='center', fontsize=11,
                fontweight='bold', zorder=3)
        ax.text(x_center + box_w / 2 + 0.35, y, desc, ha='left', va='center',
                fontsize=9, color='dimgray')

    for i in range(n - 1):
        y_from = y_positions[i] - box_h / 2
        y_to = y_positions[i + 1] + box_h / 2
        ax.annotate('', xy=(x_center, y_to), xytext=(x_center, y_from),
                    arrowprops=dict(arrowstyle='-|>', linewidth=2, color='black'))

    ax.set_title("前処理パイプライン: 生データ → モデル入力", fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(stage_dir, "preprocessing_flowchart.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 前処理フローチャート保存: {out_path}")
    return out_path


# ==============================================================================
# 3. 前処理クラス（特徴量エンジニアリング: 微分等）
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

    # ==========================================================================
    # ★追加: 前処理パイプラインのフローチャートを出力
    # ==========================================================================
    print("\n🗺  前処理フローチャートを生成します...")
    draw_preprocessing_flowchart(config.output_dir)

    # ==========================================================================
    # ★追加: 前処理パイプライン実行（位相回転 → フィルタリング → 正規化）
    #         各段階の代表信号（損傷レベルごとに3サンプル）を段階別ディレクトリへ保存
    # ==========================================================================
    print("\n📊 前処理の各段階における信号変化を可視化します...")
    stage_viz = StageSignalVisualizer(config.output_dir)
    pipeline = PreprocessingPipeline(config)
    df_train_pp, df_blind_pp = pipeline.run(df_train_full, df_blind_full, visualizer=stage_viz)
    print(f"✅ 前処理段階別プロットを {os.path.join(config.output_dir, 'preprocessing_stages')} に保存しました。\n")

    # 3. Training & Prediction Loop (L & R)
    for target_name, params in config.target_features.items():
        print("\n" + "=" * 80)
        print(f"🎯 特性 {target_name} における予測評価 (周波数: {params['f_min'] / 1e3} - {params['f_max'] / 1e3} kHz)")
        print("=" * 80)

        df_tr_curr = df_train_pp[
            (df_train_pp['[Hz]'] >= params['f_min']) & (df_train_pp['[Hz]'] <= params['f_max'])].copy()
        df_bl_curr = df_blind_pp[
            (df_blind_pp['[Hz]'] >= params['f_min']) & (df_blind_pp['[Hz]'] <= params['f_max'])].copy()

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

        # ★追加: 最終段階（特徴量エンジニアリング後 = モデル入力直前）の信号を
        #         特性(target_name)ごとのサブディレクトリに保存
        stage_viz.plot_stage(
            df_tr_fin, "train", os.path.join("05_feature_engineering", target_name),
            [c for c in config.stage_plot_cols if c in df_tr_fin.columns] + ['PR_defect'],
            n_samples=config.n_raw_samples_per_label,
            xscale='linear',
            stage_title=f"⑤ 特徴量エンジニアリング後（{target_name}帯域, {config.eng_mode}）"
        )
        stage_viz.plot_stage(
            df_bl_fin, "test", os.path.join("05_feature_engineering", target_name),
            [c for c in config.stage_plot_cols if c in df_bl_fin.columns] + ['PR_defect'],
            n_samples=config.n_raw_samples_per_label,
            xscale='linear',
            stage_title=f"⑤ 特徴量エンジニアリング後（{target_name}帯域, {config.eng_mode}）"
        )

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