import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import tensorflow as tf
import glob
import os
import shutil
import gc
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import datetime
import random
from ensemble_vis import save_training_history

# 警告を抑制
import warnings
warnings.simplefilter('ignore')

# --- 乱数シードの固定（再現性の確保） ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # GPU環境でさらに厳密な再現性を求める場合は以下も有効化します
    # tf.config.experimental.enable_op_determinism()

set_seed(42)

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
# 0. CCC Calculation Function
# ==============================================================================
def concordance_correlation_coefficient(y_true, y_pred):
    if len(y_true) < 2: return 0.0
    if np.var(y_true) < 1e-9 or np.var(y_pred) < 1e-9: return 0.0

    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    if denominator == 0: return 0.0
    return numerator / denominator

# ==============================================================================
# 1. 設定管理クラス
# ==============================================================================
class ExperimentConfig:
    def __init__(self):
        # === 共通ルートフォルダ ===
        self.base_folder = "L4"

        # === 学習用サブフォルダ ===
        self.train_folders = ["0816", "0817", "0904", "0922", "0923"]

        # === ブラインドテスト用サブフォルダ ===
        self.blind_folders = ["0121",
                              "0128"]

        # === 手動フィルタリング設定 ===
        self.manual_exclude_list = [
            "0121-V4-01",
            "0121-V5-05",
            "0128-V5-01",
            ]

        # === 特徴量・前処理設定 (上位条件に限定) ===
        self.input_modes = ["CONCAT"]
        self.rotation_strategies = [True] # 位相回転あり固定

        # ★ 数値加工: 上位2条件を採用
        self.eng_modes = ["DERIVATIVE", "L2_NORMALIZED"]

        self.model_types = ["CNN"]

        # ★ R面データのドロップ設定 (True = R面を除外)
        self.drop_r_data_options = [True]

        # ★ 周波数帯域: 1000-1500kHz のみに限定
        self.freq_ranges = [
            (1000e3, 1500e3, "1000-1500kHz"),
        ]

        # 学習パラメータ
        self.n_trials = 5
        self.patience = 15
        self.epochs = 50
        self.batch_size = 32

        # 正規化ラベル (l_map: 0.0〜1.0) を実際の損傷回数[回]に戻すための係数
        self.damage_scale_factor = 12_000_000

        # ------------------------------------------------------------------
        # ★ 追加設定: train/testを入れ替えて予測したい損傷回数一覧 (万回単位)
        #   0, 200, 400, 600, 800, 1200 の全カテゴリについて、
        #   「そのカテゴリだけをtestにして残り全部をtrainにする」学習を行う。
        #   (200, 600 は元々のブラインドテスト条件、
        #    0, 400, 800, 1200 が今回追加された条件)
        # ------------------------------------------------------------------
        self.target_damage_values_man = [0, 200, 400, 600, 800, 1200]

        # ブラインドフォルダ内の未知歯 (V4/V5) の真の損傷回数 [回]
        # ファイル名(condition)に含まれる 'v4' / 'v5' の文字列から判定する。
        self.blind_ground_truth = {
            "v4": 2_000_000,
            "v5": 6_000_000,
        }

        # 結果出力先のルートフォルダ
        # ★ 各損傷回数(万回)ごとに ./result/distribution/{損傷回数} に保存する
        self.output_root = "./result/distribution"

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

    def load_train_data(self):
        print("📥 Loading Training Data...")
        return self._load_data_generic(self.cfg.train_folders, is_blind=False)

    def load_blind_data(self):
        print("📥 Loading Blind Data...")
        return self._load_data_generic(self.cfg.blind_folders, is_blind=True)

    def _load_data_generic(self, folders, is_blind=False):
        air_paths, data_paths = self._collect_file_paths(folders)
        if not data_paths:
            print(f"⚠ No data found in specified folders under {self.cfg.base_folder}")
            return None

        air_avg = self._create_air_average(air_paths)
        data_dicts = self._apply_air_normalization(data_paths, air_avg)
        df_paired = self._process_strict_fr_rescue(data_dicts, is_blind)

        if df_paired is not None and not df_paired.empty:
            if not is_blind:
                df_paired = df_paired[df_paired['label'] != 'damaged_pitching'].copy()

            def parse_condition(cond):
                if "_" in cond:
                    parts = cond.split('_', 1)
                    if len(parts) > 1 and len(parts[0]) == 4 and parts[0].isdigit():
                        real_fname = parts[1]
                    else:
                        real_fname = cond
                else:
                    real_fname = cond

                if len(real_fname) >= 12:
                    return real_fname[5:7], real_fname[8:10]
                return "Unk", "Unk"

            meta = df_paired['condition'].apply(lambda x: pd.Series(parse_condition(x), index=['GearID', 'ToothID']))
            df_paired = pd.concat([df_paired, meta], axis=1)

            if self.cfg.manual_exclude_list:
                df_paired['filter_key'] = df_paired['folder'] + "-" + df_paired['GearID'] + "-" + df_paired['ToothID']
                df_paired = df_paired[~df_paired['filter_key'].isin(self.cfg.manual_exclude_list)].copy()

            df_paired['group_key'] = df_paired['folder'] + "_" + df_paired['GearID'] + "_" + df_paired['ToothID']

        return df_paired

    def _collect_file_paths(self, folders):
        air_paths = []
        data_paths = []
        for folder in folders:
            folder_path = os.path.join(self.cfg.base_folder, folder)
            if not os.path.exists(folder_path):
                print(f"⚠ Folder not found: {folder_path}")
                continue

            air_paths.extend(glob.glob(os.path.join(folder_path, "*air*.csv")))
            for f in glob.glob(os.path.join(folder_path, "*.csv")):
                fname = os.path.basename(f)
                if "air" in fname.lower(): continue
                data_paths.append(f)
        return air_paths, data_paths

    def _create_air_average(self, paths):
        avg_dict = {}
        temp_dict = defaultdict(list)
        cols = ['[Hz]', 'LS[H]', '|Z|[ohm]', 'RS[ohm]', 'Q(s)[-]', 'X[ohm]', 'PhaseZ[deg]']
        for p in paths:
            folder = os.path.basename(os.path.dirname(p))
            try:
                df = pd.read_csv(p, encoding='utf-8-sig')[cols].dropna()
                temp_dict[folder].append(df)
            except: continue
        for folder, dfs in temp_dict.items():
            if dfs:
                combined = pd.concat(dfs)
                avg = combined.groupby('[Hz]', as_index=False).mean()
                avg.rename(columns={'LS[H]':'L0','|Z|[ohm]':'Z0','RS[ohm]':'R0','Q(s)[-]':'Q0','X[ohm]':'X0','PhaseZ[deg]':'PhaseZ0'}, inplace=True)
                avg_dict[folder] = avg
        return avg_dict

    def _apply_air_normalization(self, paths, air_avg):
        processed = {}
        mapping = {'LS[H]':'L0','|Z|[ohm]':'Z0','RS[ohm]':'R0','Q(s)[-]':'Q0','X[ohm]':'X0','PhaseZ[deg]':'PhaseZ0'}
        for p in paths:
            folder = os.path.basename(os.path.dirname(p))
            fname = os.path.splitext(os.path.basename(p))[0]
            if folder not in air_avg: continue
            try:
                df = pd.read_csv(p, encoding='utf-8-sig')
                df = pd.merge(df, air_avg[folder], on='[Hz]', how='left')
                for col, air_col in mapping.items():
                    if col in df.columns and air_col in df.columns:
                        df[f"{col}_airdiff"] = df[col] - df[air_col]
                df['folder'] = folder
                df['original_filename'] = fname
                processed[(folder, fname)] = df
            except: continue
        return processed

    def _process_strict_fr_rescue(self, data_dicts, is_blind):
        target_cols = ['LS[H]_airdiff', '|Z|[ohm]_airdiff', 'RS[ohm]_airdiff', 'Q(s)[-]_airdiff', 'X[ohm]_airdiff', 'PhaseZ[deg]_airdiff']
        groups = defaultdict(list)

        for (folder, fname), df in data_dicts.items():
            if len(fname) < 10: continue
            cond_code = fname[2:5]
            gear_id = fname[5:7]
            side = fname[7]
            tooth_id = fname[8:10]
            group_key = (folder, gear_id, tooth_id)

            df_temp = df.copy()
            df_temp['side'] = side
            if is_blind:
                map_search_key = "Reference" if "000N" in cond_code else "Blind"
            else:
                map_search_key = cond_code + gear_id + side

            df_temp['map_key'] = map_search_key
            groups[group_key].append((fname, df_temp))

        all_data = []
        r_reference_bank = defaultdict(list)
        valid_groups = []
        rescue_groups = []

        for group_key, items in groups.items():
            has_R = any(df['side'].iloc[0] == 'R' for _, df in items)
            if has_R: valid_groups.append((group_key, items))
            else: rescue_groups.append((group_key, items))

        for group_key, items in valid_groups:
            folder, gear_id, tooth_id = group_key
            R_dfs_tuple = [(name, df) for name, df in items if df['side'].iloc[0] == 'R']
            F_dfs_tuple = [(name, df) for name, df in items if df['side'].iloc[0] == 'F']

            R_dfs_only = [df for name, df in R_dfs_tuple]
            R_combined = pd.concat(R_dfs_only)
            R_mean = R_combined.groupby('[Hz]', as_index=False)[target_cols].mean()
            r_reference_bank[(gear_id, tooth_id)].append(R_mean)

            for fname_f, df_f in F_dfs_tuple:
                merged = pd.merge(df_f, R_mean, on='[Hz]', suffixes=('_Sig', '_Ref'))
                unique_fname = f"{folder}_{fname_f}"
                final_df = self._finalize_df_cols(merged, unique_fname, df_f['map_key'].iloc[0], target_cols, 'F', is_blind)
                all_data.append(final_df)

            for fname_r, df_r in R_dfs_tuple:
                merged = pd.merge(df_r, R_mean, on='[Hz]', suffixes=('_Sig', '_Ref'))
                unique_fname = f"{folder}_{fname_r}"
                final_df = self._finalize_df_cols(merged, unique_fname, "Reference", target_cols, 'R', is_blind)
                all_data.append(final_df)

        for group_key, items in rescue_groups:
            folder, gear_id, tooth_id = group_key
            F_dfs_tuple = [(name, df) for name, df in items if df['side'].iloc[0] == 'F']
            if not F_dfs_tuple: continue

            available_refs = r_reference_bank.get((gear_id, tooth_id))
            if not available_refs: continue
            grand_ref_combined = pd.concat(available_refs)
            Grand_R_mean = grand_ref_combined.groupby('[Hz]', as_index=False)[target_cols].mean()

            for fname_f, df_f in F_dfs_tuple:
                merged = pd.merge(df_f, Grand_R_mean, on='[Hz]', suffixes=('_Sig', '_Ref'))
                unique_fname = f"{folder}_{fname_f}"
                final_df = self._finalize_df_cols(merged, unique_fname, df_f['map_key'].iloc[0], target_cols, 'F', is_blind)
                final_df['is_rescued'] = True
                all_data.append(final_df)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def _finalize_df_cols(self, df, fname, map_key, target_cols, side, is_blind):
        sig_cols = [f"{c}_Sig" for c in target_cols]
        ref_cols = [f"{c}_Ref" for c in target_cols]
        keep_cols = sig_cols + ref_cols + ['[Hz]', 'folder']
        out = df[keep_cols].copy()
        out['condition'] = fname
        out['side'] = side

        if is_blind:
            if map_key == "Reference":
                out['label'] = 'no_damaged'
            else:
                out['label'] = 'Unknown'
        else:
            if map_key == "Reference":
                out['label'] = 'no_damaged'
            else:
                out['label'] = self.damage_map.get(map_key, "no_label")
        return out

# ==============================================================================
# 3. 前処理クラス
# ==============================================================================
class StrictPreprocessor:
    def __init__(self, config, input_mode, use_rotation):
        self.cfg = config
        self.input_mode = input_mode
        self.use_rotation = use_rotation

    def fit(self, df_train):
        return self

    def transform(self, df):
        df_proc = df.copy()
        if self.use_rotation:
            df_proc = self._apply_day_rotation(df_proc)
        df_out = self._recalculate_parameters(df_proc)
        return df_out

    def _apply_day_rotation(self, df):
        rotated_dfs = []
        suffixes = ['_Sig', '_Ref']

        for folder, group in df.groupby('folder'):
            ref_data = group[group['label'] == 'no_damaged']
            if ref_data.empty:
                rotated_dfs.append(group)
                continue

            ref_means = ref_data.groupby('[Hz]')[['RS[ohm]_airdiff_Sig', 'X[ohm]_airdiff_Sig']].mean()
            current_angles = np.arctan2(
                ref_means['X[ohm]_airdiff_Sig'].values,
                ref_means['RS[ohm]_airdiff_Sig'].values
            )
            rot_angles = (np.pi / 2) - current_angles
            angle_df = pd.DataFrame({'[Hz]': ref_means.index, 'rot_angle': rot_angles})

            group_merged = pd.merge(group, angle_df, on='[Hz]', how='left')
            theta = group_merged['rot_angle'].values
            c = np.cos(theta)
            s = np.sin(theta)

            group_rot = group.copy()
            for suff in suffixes:
                r_col = f'RS[ohm]_airdiff{suff}'
                x_col = f'X[ohm]_airdiff{suff}'
                if r_col in group_rot.columns and x_col in group_rot.columns:
                    r_vals = group_rot[r_col].values
                    x_vals = group_rot[x_col].values
                    group_rot[r_col] = r_vals * c - x_vals * s
                    group_rot[x_col] = r_vals * s + x_vals * c
            rotated_dfs.append(group_rot)
        return pd.concat(rotated_dfs)

    def _recalculate_parameters(self, df):
        # ★ "true_value" (真の損傷回数[回]) をメタ情報として保持する
        meta_cols = ["condition", "label", "folder", "[Hz]", "side", "GearID", "ToothID",
                     "group_key", "is_rescued", "map_key", "true_value"]
        meta_cols = [c for c in meta_cols if c in df.columns]
        df_out = df[meta_cols].copy()

        freq = df['[Hz]'].values

        def calc_physical_params(r_vec, x_vec, suffix_out=""):
            z = np.sqrt(r_vec**2 + x_vec**2)
            ph = np.degrees(np.arctan2(x_vec, r_vec))
            l = x_vec / (2 * np.pi * freq)
            q = x_vec / (r_vec + 1e-9)
            return {
                f'LS[H]_airdiff{suffix_out}': l,
                f'|Z|[ohm]_airdiff{suffix_out}': z,
                f'RS[ohm]_airdiff{suffix_out}': r_vec,
                f'Q(s)[-]_airdiff{suffix_out}': q,
                f'X[ohm]_airdiff{suffix_out}': x_vec,
                f'PhaseZ[deg]_airdiff{suffix_out}': ph
            }

        r_sig = df['RS[ohm]_airdiff_Sig'].values
        x_sig = df['X[ohm]_airdiff_Sig'].values
        r_ref = df['RS[ohm]_airdiff_Ref'].values
        x_ref = df['X[ohm]_airdiff_Ref'].values
        r_calib = r_sig - r_ref
        x_calib = x_sig - x_ref

        if self.input_mode == "CALIB":
            res = calc_physical_params(r_calib, x_calib, "")
            for k, v in res.items(): df_out[k] = v
        elif self.input_mode == "RAW":
            res = calc_physical_params(r_sig, x_sig, "")
            for k, v in res.items(): df_out[k] = v
        elif self.input_mode == "CONCAT":
            res_sig = calc_physical_params(r_sig, x_sig, "_Sig")
            for k, v in res_sig.items(): df_out[k] = v
            res_ref = calc_physical_params(r_ref, x_ref, "_Ref")
            for k, v in res_ref.items(): df_out[k] = v
            res_calib = calc_physical_params(r_calib, x_calib, "_Calib")
            for k, v in res_calib.items(): df_out[k] = v
        return df_out

    def apply_feature_engineering(self, df_input, eng_mode):
        exclude = ["condition", "label", "folder", "[Hz]", "side", "GearID", "ToothID",
                   "group_key", "is_rescued", "map_key", "true_value"]
        feature_cols = [c for c in df_input.columns if c not in exclude]

        df_sorted = df_input.sort_values(by=['condition', '[Hz]'])
        processed_dfs = []
        try:
            pd.DataFrame().groupby('a').apply(lambda x: x, include_groups=False)
            kwargs = {'include_groups': False}
        except: kwargs = {}

        # ★ true_value が存在する場合はメタ情報として保持する
        base_meta_cols = ['condition', 'label', 'folder', '[Hz]']
        if 'true_value' in df_sorted.columns:
            base_meta_cols = base_meta_cols + ['true_value']

        for name, group in df_sorted.groupby('condition'):
            g = group.copy()
            g_res = g[base_meta_cols].copy()
            for col in feature_cols:
                vals = g[col].values
                if eng_mode == "DERIVATIVE":
                    g_res[col] = np.gradient(vals, g['[Hz]'].values)
                elif eng_mode == "L2_NORMALIZED":
                    norm = np.sqrt(np.sum(vals**2))
                    if norm > 0: g_res[col] = vals / norm
                    else: g_res[col] = vals
                else:
                    g_res[col] = vals
            processed_dfs.append(g_res)
        return pd.concat(processed_dfs)

    def generate_sequences(self, df, is_blind=False):
        """
        is_blind=False: 学習用シーケンスを生成する。y は 'true_value'[回] を
                        config.damage_scale_factor で正規化した値になる。
        is_blind=True : 推論専用のブラインド/ターゲットデータのシーケンスを
                        生成する。y は使用しないため 0 埋めとする
                        (真の損傷回数は 'true_value' 列 / conditions 経由で
                        別途参照可能)。
        """
        sequences = []
        labels = []
        conditions = []
        true_values = []

        exclude = ["condition", "label", "folder", "[Hz]", "side", "GearID", "ToothID",
                   "group_key", "is_rescued", "map_key", "true_value"]
        features = [c for c in df.columns if c not in exclude]

        for name, group in df.groupby('condition'):
            if group.empty: continue
            seq = group[features].values
            sequences.append(seq)
            labels.append(group['label'].iloc[0])
            conditions.append(name)
            if 'true_value' in group.columns:
                true_values.append(group['true_value'].iloc[0])
            else:
                true_values.append(np.nan)

        X = np.array(sequences)
        if is_blind:
            y = np.zeros(len(labels))
        else:
            scale = self.cfg.damage_scale_factor
            y = np.array([
                (tv / scale) if (tv is not None and not np.isnan(tv)) else 0.0
                for tv in true_values
            ])
        return X, y, conditions, features

# ==============================================================================
# 4. モデル学習クラス
# ==============================================================================
class ModelTrainer:
    def __init__(self, config):
        self.cfg = config

    def build_model(self, input_shape, model_type="ANN"):
        model = Sequential()
        if model_type == "CNN":
            model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(64, kernel_size=3, activation='relu'))
            model.add(GlobalAveragePooling1D())
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
        else:
            model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
            model.add(Dropout(0.2))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        return model

    def train_full_and_predict(self, X_train, y_train, X_blind, model_type):
        scaler = StandardScaler()

        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        if model_type == "CNN":
            N_tr, T, F = X_tr.shape
            X_tr_flat = X_tr.reshape(N_tr, -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            X_bl_flat = X_blind.reshape(X_blind.shape[0], -1)

            X_tr_sc = scaler.fit_transform(X_tr_flat).reshape(N_tr, T, F)
            X_val_sc = scaler.transform(X_val_flat).reshape(X_val.shape[0], T, F)
            X_bl_sc = scaler.transform(X_bl_flat).reshape(X_blind.shape[0], T, F)
            input_shape = (T, F)
        else: # ANN
            N_tr = X_tr.shape[0]
            X_tr_flat = X_tr.reshape(N_tr, -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            X_bl_flat = X_blind.reshape(X_blind.shape[0], -1)

            X_tr_sc = scaler.fit_transform(X_tr_flat)
            X_val_sc = scaler.transform(X_val_flat)
            X_bl_sc = scaler.transform(X_bl_flat)
            input_shape = X_tr_sc.shape[1]

        model = self.build_model(input_shape, model_type)
        history = model.fit(X_tr_sc, y_tr, validation_data=(X_val_sc, y_val),
                            epochs=self.cfg.epochs, batch_size=self.cfg.batch_size, verbose=0,
                            callbacks=[EarlyStopping(patience=self.cfg.patience, restore_best_weights=True)])

        y_pred_tr = model.predict(X_tr_sc, verbose=0).flatten()
        y_pred_val = model.predict(X_val_sc, verbose=0).flatten()
        y_pred_blind_raw = model.predict(X_bl_sc, verbose=0).flatten()

        scale_factor = self.cfg.damage_scale_factor
        val_rmse = np.sqrt(mean_squared_error(y_val * scale_factor, y_pred_val * scale_factor))
        val_ccc = concordance_correlation_coefficient(y_val * scale_factor, y_pred_val * scale_factor)

        result = {
            "preds_inf": y_pred_blind_raw,
            "y_val_true": y_val * scale_factor,
            "y_val_pred": y_pred_val * scale_factor,
            "val_ccc": val_ccc,
            "history": history
        }

        # ------------------------------------------------------------------
        # ★ メモリ対策: モデル・中間変数を明示的に破棄し、Kerasのセッションを
        #    クリアする。TensorFlow/Kerasは学習を繰り返すたびにグラフや
        #    変数がプロセス内に蓄積されやすく、これがメモリオーバーの主因に
        #    なりやすいため、1回の学習が終わるごとに必ず解放する。
        # ------------------------------------------------------------------
        del model, X_tr_sc, X_val_sc, X_bl_sc, X_tr, X_val, y_tr, y_val
        tf.keras.backend.clear_session()
        gc.collect()

        return result

# ==============================================================================
# 5. Hybrid Ensemble Aggregator Class (Weighted Averaging)
# ==============================================================================
class HybridEnsembleAggregator:
    def __init__(self, output_dir, scale_factor):
        self.output_dir = output_dir
        self.trial_results = []
        self.file_names = None
        self.SCALE_FACTOR = float(scale_factor)

    def add_trial_result(self, unique_id, val_ccc, preds, file_names):
        """
        1回の試行結果を追加する
        unique_id: "CNN_CONCAT_ROT_DERIVATIVE_1000-1500kHz" のような識別子
        """
        self.trial_results.append({
            'id': unique_id,
            'val_ccc': val_ccc,
            'preds': preds
        })
        if self.file_names is None:
            self.file_names = file_names

    def run_ensemble(self):
        """
        検証スコア(CCC)の比率に基づいて重み付け平均を行うアンサンブル
        """
        if not self.trial_results:
            print("No results to ensemble.")
            return

        # ディレクトリ作成
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # ---------------------------------------------------------
        # 1. 結果の整理とグループ化
        # ---------------------------------------------------------
        grouped_results = defaultdict(list)

        for res in self.trial_results:
            parts = res['id'].split('_')
            if len(parts) >= 4:
                eng_mode = parts[3] # DERIVATIVE or L2_NORMALIZED
            else:
                eng_mode = "Unknown"

            grouped_results[eng_mode].append(res)

        # ---------------------------------------------------------
        # 2. 条件ごとの平均スコアと平均予測値の算出
        # ---------------------------------------------------------
        condition_summaries = {}
        total_ccc_score = 0.0

        print("\n=== Condition Summary ===")
        for eng_mode, results in grouped_results.items():
            preds_matrix = np.array([r['preds'] for r in results])

            mean_preds_raw = np.mean(preds_matrix, axis=0)
            mean_preds_scaled = mean_preds_raw * self.SCALE_FACTOR

            mean_ccc = np.mean([r['val_ccc'] for r in results])

            if mean_ccc < 0: mean_ccc = 0.0

            condition_summaries[eng_mode] = {
                'mean_preds': mean_preds_scaled,
                'mean_ccc': mean_ccc,
                'std_preds': np.std(preds_matrix * self.SCALE_FACTOR, axis=0)
            }
            total_ccc_score += mean_ccc
            print(f"Condition: {eng_mode}, Mean CCC: {mean_ccc:.4f}, Trials: {len(results)}")

        # ---------------------------------------------------------
        # 3. 重み付け平均による最終予測
        # ---------------------------------------------------------
        if total_ccc_score == 0:
            print("⚠ Total CCC score is 0. Cannot compute weighted average.")
            return

        n_samples = len(self.file_names)
        final_preds = np.zeros(n_samples)

        print("\n=== Weighted Ensemble Calculation ===")

        weights_info = []
        weight_rows = []
        for eng_mode, summary in condition_summaries.items():
            weight = summary['mean_ccc'] / total_ccc_score
            weight_rows.append({
                "target_damage_man": os.path.basename(os.path.normpath(self.output_dir)),
                "eng_mode": eng_mode,
                "mean_validation_CCC": summary["mean_ccc"],
                "ensemble_weight": weight,
            })
            weighted_preds = summary['mean_preds'] * weight
            final_preds += weighted_preds

            weights_info.append(f"{eng_mode}: Weight={weight:.3f} (CCC={summary['mean_ccc']:.3f})")

        pd.DataFrame(weight_rows).to_csv(
            os.path.join(self.output_dir, "Condition_Weights.csv"),
            index=False, encoding="utf-8-sig"
        )

        print("Weights used:", ", ".join(weights_info))

        # ---------------------------------------------------------
        # 4. 信頼区間 (±2sigma) の算出
        # ---------------------------------------------------------
        final_std = np.zeros(n_samples)
        for eng_mode, summary in condition_summaries.items():
            weight = summary['mean_ccc'] / total_ccc_score
            final_std += summary['std_preds'] * weight

        lower_bound = final_preds - 2 * final_std
        upper_bound = final_preds + 2 * final_std

        final_preds = np.maximum(final_preds, 0)
        lower_bound = np.maximum(lower_bound, 0)
        upper_bound = np.maximum(upper_bound, 0)

        # ---------------------------------------------------------
        # 5. 結果出力
        # ---------------------------------------------------------
        df_result = pd.DataFrame({
            "FileName": self.file_names,
            "Predicted_Damage": final_preds,
            "Lower_2sigma": lower_bound,
            "Upper_2sigma": upper_bound,
            "Std_Dev": final_std
        })

        try:
            df_result['GearID'] = df_result['FileName'].apply(lambda x: x.split('_')[1][5:7])
            df_result['ToothID'] = df_result['FileName'].apply(lambda x: x.split('_')[1][8:10])
            df_result = df_result.sort_values(by=['GearID', 'ToothID'])
            df_result = df_result.drop(columns=['GearID', 'ToothID'])
        except:
            pass

        out_path = os.path.join(self.output_dir, "Final_Weighted_Ensemble_Result.csv")
        df_result.to_csv(out_path, index=False)
        print(f"\n✅ Final weighted ensemble results saved to: {out_path}")
        print(df_result.head())

# ==============================================================================
# 6. 統合データセット構築 (train + blind を「真の損傷回数」付きで結合)
# ==============================================================================
def build_unified_dataset(df_train, df_blind, config):
    """
    学習用データ(df_train)とブラインドデータ(df_blind)を、
    それぞれの「真の損傷回数(true_value)[回]」を付与した上で
    1つのデータフレームに結合する。

    - df_train 側: label (no_damaged / damaged_040 / damaged_080 / damaged_800)
      から damage_scale_factor を用いて実回数に変換 (0, 400万, 800万, 1200万回)。
    - df_blind 側: label=='no_damaged' の基準データは 0回。
      それ以外(Unknown)は condition(ファイル名)に含まれる 'v4' / 'v5' の
      文字列から config.blind_ground_truth の値 (200万回 / 600万回) を割り当てる。
      どちらにも該当しない行は true_value=NaN とし、後段で除外する。
    """
    df_train = df_train.copy()
    df_blind = df_blind.copy()

    scale = config.damage_scale_factor
    l_map_frac = {'no_damaged': 0.0, 'damaged_040': 1/3, 'damaged_080': 2/3, 'damaged_800': 1.0}
    df_train['true_value'] = df_train['label'].map(l_map_frac) * scale

    gt_map = config.blind_ground_truth  # e.g. {"v4": 2_000_000, "v5": 6_000_000}

    def _blind_true_value(row):
        if row['label'] == 'no_damaged':
            return 0.0
        cond = str(row['condition']).lower()
        for key, val in gt_map.items():
            if key.lower() in cond:
                return float(val)
        return np.nan

    df_blind['true_value'] = df_blind.apply(_blind_true_value, axis=1)

    df_all = pd.concat([df_train, df_blind], ignore_index=True)

    n_before = len(df_all)
    df_all = df_all.dropna(subset=['true_value']).copy()
    n_after = len(df_all)
    if n_before != n_after:
        print(f"ℹ 真の損傷回数が判定できなかった {n_before - n_after} 行を除外しました。")

    return df_all

# ==============================================================================
# 7. ターゲット損傷回数ごとの実験実行 (train/testの入れ替え)
# ==============================================================================
def run_single_target_experiment(target_man, df_all, config, trainer):
    """
    target_man: 損傷回数 [万回] (例: 0, 200, 400, 600, 800, 1200)

    df_all の中から true_value == target の行を test(ブラインド予測対象)、
    それ以外を train として扱い、既存のCNNアンサンブル学習パイプラインを
    そのまま実行する。結果は ./result/distribution/{target_man} に保存する。
    """
    target_value = target_man * 10_000  # 万回 -> 回
    output_dir = os.path.join(config.output_root, str(target_man))
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"👉 Target damage count: {target_man}万回 ({target_value:,.0f} cycles)")
    print(f"{'='*70}")

    df_test_target = df_all[np.isclose(df_all['true_value'], target_value)].copy()
    df_train_rest = df_all[~np.isclose(df_all['true_value'], target_value)].copy()

    if df_test_target.empty:
        print(f"⚠ 損傷回数 {target_man}万回 に該当するデータが見つかりません。スキップします。")
        return
    if df_train_rest.empty:
        print(f"⚠ ターゲットを除外した後の学習データが空です。スキップします。")
        return

    aggregator = HybridEnsembleAggregator(output_dir, config.damage_scale_factor)

    for model_type in config.model_types:
        for input_mode in config.input_modes:
            for use_rot in config.rotation_strategies:
                for eng_mode in config.eng_modes:
                    for drop_r in config.drop_r_data_options:
                        for (f_min, f_max, f_label) in config.freq_ranges:

                            unique_id = f"{model_type}_{input_mode}_{'ROT' if use_rot else 'NOROT'}_{eng_mode}_{f_label}"
                            if drop_r: unique_id += "_NoR"
                            print(f"\n   [{target_man}万回] Processing: {unique_id}")

                            df_tr_curr = df_train_rest[
                                (df_train_rest['[Hz]'] >= f_min) & (df_train_rest['[Hz]'] <= f_max)
                            ].copy()
                            df_te_curr = df_test_target[
                                (df_test_target['[Hz]'] >= f_min) & (df_test_target['[Hz]'] <= f_max)
                            ].copy()

                            if drop_r:
                                print("      ℹ Filtering out 'R' side data...")
                                if 'side' in df_tr_curr.columns:
                                    df_tr_curr = df_tr_curr[df_tr_curr['side'] == 'F'].copy()
                                if 'side' in df_te_curr.columns:
                                    df_te_curr = df_te_curr[df_te_curr['side'] == 'F'].copy()

                            if df_tr_curr.empty or df_te_curr.empty:
                                print("      ⚠ Empty dataframe after filtering. Skipping.")
                                continue

                            for trial in range(config.n_trials):
                                print(f"      [Trial {trial+1}/{config.n_trials}]")

                                preproc = StrictPreprocessor(config, input_mode, use_rot)

                                df_tr_trans = preproc.transform(df_tr_curr)
                                df_te_trans = preproc.transform(df_te_curr)

                                df_tr_fin = preproc.apply_feature_engineering(df_tr_trans, eng_mode)
                                df_te_fin = preproc.apply_feature_engineering(df_te_trans, eng_mode)

                                X_tr, y_tr, _, _ = preproc.generate_sequences(df_tr_fin, is_blind=False)
                                X_te, _, te_conds, _ = preproc.generate_sequences(df_te_fin, is_blind=True)

                                if len(X_tr) == 0 or len(X_te) == 0: continue

                                res = trainer.train_full_and_predict(X_tr, y_tr, X_te, model_type)
                                save_training_history(
                                    res["history"], config.output_root, target_man, eng_mode,
                                    trial + 1, unique_id
                                )

                                aggregator.add_trial_result(
                                    unique_id,
                                    res['val_ccc'],
                                    res['preds_inf'],
                                    te_conds
                                )

                                # ★ trial単位の中間データも都度解放しておく
                                del df_tr_trans, df_te_trans, df_tr_fin, df_te_fin, X_tr, y_tr, X_te, res
                                gc.collect()

    aggregator.run_ensemble()
    visualize_single_target(output_dir, target_value, target_man)

    # ★ ターゲット1件が終わったら、そのデータフレーム群も明示的に解放する
    del df_test_target, df_train_rest, aggregator
    gc.collect()


def visualize_single_target(output_dir: str, target_value: float, target_man):
    """
    Final_Weighted_Ensemble_Result.csv を読み込み、
    予測損傷回数のヒストグラムを描画する。正解値 (target_value) は
    赤の垂直線で表示する。ビン幅は 50万回 (500,000) 固定。
    """
    csv_path = os.path.join(output_dir, "Final_Weighted_Ensemble_Result.csv")
    if not os.path.exists(csv_path):
        print(f"⚠ CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if "Predicted_Damage" not in df.columns:
        print(f"⚠ Column 'Predicted_Damage' not found in CSV.")
        return

    df["Predicted_Damage"] = pd.to_numeric(df["Predicted_Damage"], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["Predicted_Damage"])

    print("\n" + "=" * 60)
    print(f"  未知歯車 損傷回数推定サマリー ({target_man}万回)")
    print("=" * 60)

    n = len(df)
    if n == 0:
        print("   データなし")
        print()
        return

    preds = df["Predicted_Damage"].values
    mean_pred = np.mean(preds)
    std_pred = np.std(preds)
    print(f"正解値       : {target_value:,.0f} 回")
    print(f"データ数     : {n}")
    print(f"予測損傷回数 : {mean_pred:,.1f} 回  (平均)")
    print(f"標準偏差     : {std_pred:,.1f} 回")
    print("-" * 60 + "\n")

    BIN_WIDTH = 500_000
    data_min = min(preds.min(), target_value)
    data_max = max(preds.max(), target_value)
    range_min = max(0.0, np.floor(data_min / BIN_WIDTH) * BIN_WIDTH - BIN_WIDTH)
    range_max = np.ceil(data_max / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH
    bin_edges = np.arange(range_min, range_max + BIN_WIDTH, BIN_WIDTH)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(preds, bins=bin_edges, color="steelblue", alpha=0.75, edgecolor='white',
            label=f"Predicted (n={n})")
    ax.axvline(target_value, color='red', linewidth=2.5, linestyle='--',
               label=f"Ground Truth: {target_value:,.0f}")
    ax.axvline(mean_pred, color='black', linewidth=1.8, linestyle=':',
               label=f"Mean Pred: {mean_pred:,.0f}")

    stats_text = f"n = {n}\nMean = {mean_pred:,.0f}\nStd  = {std_pred:,.0f}"
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    ax.set_title(f"Blind Test — Predicted Damage Count Distribution ({target_man}万回)",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Predicted Damage Count (cycles)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_xlim(range_min, range_max)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', rotation=20)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Histogram.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Histogram saved to: {save_path}")


# ==============================================================================
# 8. 対話形式でのターゲット選択
# ==============================================================================
def prompt_target_selection(config):
    """
    config.target_damage_values_man の中から、今回処理したい損傷回数(万回)を
    1つだけユーザーに選んでもらう。全部一気に実行するとメモリオーバーで
    落ちるため、必ず1つずつ選択して実行する。
    """
    options = config.target_damage_values_man
    print("\n--- 予測したい損傷回数(万回)を選択してください ---")
    for i, val in enumerate(options, start=1):
        print(f"  [{i}] {val} 万回")
    print("  [q] 終了")

    while True:
        choice = input("番号または損傷回数(万回)を入力してください: ").strip()

        if choice.lower() in ("q", "quit", "exit"):
            return None

        # 番号での指定 (例: "1")
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
            # 番号として範囲外でも、そのまま損傷回数(万回)として解釈できるか試す
            if idx in options:
                return idx

        # 損傷回数そのものでの指定 (例: "200")
        try:
            val = int(choice)
            if val in options:
                return val
        except ValueError:
            pass

        print(f"⚠ 入力が正しくありません。番号(1-{len(options)})か、"
              f"損傷回数({options})のいずれかを入力してください。")


# ==============================================================================
# 9. Main Execution Flow (対話形式・1つずつ実行)
# ==============================================================================
def main():
    print("🚀 Starting Distribution-Swap Ensemble Process (Interactive Mode)...")

    config = ExperimentConfig()
    data_mgr = DataManager(config)
    trainer = ModelTrainer(config)

    # 1. Load Data (元々のtrainフォルダとblindフォルダを両方読み込む)
    #    ※ データ読み込みは重い処理なので、対話ループの外(=1回だけ)で行う
    df_train_full = data_mgr.load_train_data()
    df_blind_full = data_mgr.load_blind_data()

    if df_train_full is None or df_blind_full is None or df_train_full.empty or df_blind_full.empty:
        print("❌ Data loading failed.")
        return

    # 2. train/blind を「真の損傷回数」付きで統合
    df_all = build_unified_dataset(df_train_full, df_blind_full, config)
    if df_all.empty:
        print("❌ 統合データセットが空です。真の損傷回数の割り当てを確認してください。")
        return

    print("\n=== Unified Dataset Summary (conditionファイル数 / 真の損傷回数別) ===")
    summary = df_all.drop_duplicates(subset=['condition'])['true_value'].value_counts().sort_index()
    print(summary)

    os.makedirs(config.output_root, exist_ok=True)

    # 元の読み込み済みデータフレームはもう不要なので解放
    del df_train_full, df_blind_full
    gc.collect()

    # 3. 対話形式で「1つの損傷回数」だけを選び、その回数だけをtestに、
    #    残り全部をtrainにして学習・予測する。終わったら続けるか確認する。
    while True:
        target_man = prompt_target_selection(config)
        if target_man is None:
            print("終了します。")
            break

        run_single_target_experiment(target_man, df_all, config, trainer)

        # ★ 1ターゲット分の学習・推論が終わるたびにメモリを解放
        gc.collect()

        cont = input("\n続けて別の損傷回数を処理しますか？ (y/n): ").strip().lower()
        if cont not in ("y", "yes"):
            print("終了します。")
            break

    print("\n✅ Finished.")


if __name__ == "__main__":
    main()