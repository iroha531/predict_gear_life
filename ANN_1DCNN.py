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
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import datetime

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
        self.base_folder = "./dataset/L4"

        # === 学習用サブフォルダ ===
        self.train_folders = ["train"]
        
        # === ブラインドテスト用サブフォルダ ===
        self.blind_folders = ["test"]
        
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
        
        # 結果出力先
        self.output_dir = "./result/ensemble_predictions" 

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
        meta_cols = ["condition", "label", "folder", "[Hz]", "side", "GearID", "ToothID", "group_key", "is_rescued", "map_key"]
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
        exclude = ["condition", "label", "folder", "[Hz]", "side", "GearID", "ToothID", "group_key", "is_rescued", "map_key"]
        feature_cols = [c for c in df_input.columns if c not in exclude]
        
        df_sorted = df_input.sort_values(by=['condition', '[Hz]'])
        processed_dfs = []
        try:
            pd.DataFrame().groupby('a').apply(lambda x: x, include_groups=False)
            kwargs = {'include_groups': False}
        except: kwargs = {}

        for name, group in df_sorted.groupby('condition'):
            g = group.copy()
            g_res = g[['condition', 'label', 'folder', '[Hz]']].copy()
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
        sequences = []
        labels = []
        conditions = []
        
        exclude = ["condition", "label", "folder", "[Hz]", "side", "GearID", "ToothID", "group_key", "is_rescued", "map_key"]
        features = [c for c in df.columns if c not in exclude]
        
        for name, group in df.groupby('condition'):
            if group.empty: continue
            seq = group[features].values
            sequences.append(seq)
            labels.append(group['label'].iloc[0])
            conditions.append(name)
            
        X = np.array(sequences)
        if is_blind:
            y = np.zeros(len(labels))
        else:
            l_map = {'no_damaged': 0.0, 'damaged_040': 1/3, 'damaged_080': 2/3, 'damaged_800': 1.0}
            y = np.array([l_map.get(l, 0.0) for l in labels])
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
        
        scale_factor = 1200
        val_rmse = np.sqrt(mean_squared_error(y_val * scale_factor, y_pred_val * scale_factor))
        val_ccc = concordance_correlation_coefficient(y_val * scale_factor, y_pred_val * scale_factor)
        
        return {
            "preds_inf": y_pred_blind_raw,
            "y_val_true": y_val * scale_factor,
            "y_val_pred": y_pred_val * scale_factor,
            "val_ccc": val_ccc,
            "history": history
        }

# ==============================================================================
# 5. Hybrid Ensemble Aggregator Class (Weighted Averaging)
# ==============================================================================
class HybridEnsembleAggregator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.trial_results = []
        self.file_names = None
        self.SCALE_FACTOR = 1200.0

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
        # IDから条件キー（数値加工法など）を抽出してグループ化
        # 例: id="CNN_CONCAT_ROT_DERIVATIVE_1000-1500kHz" -> key="DERIVATIVE"
        grouped_results = defaultdict(list)
        
        for res in self.trial_results:
            parts = res['id'].split('_')
            # 命名規則: [Model, Input, Rot, EngMode, Freq]
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
            # 予測値行列: (N_trials, N_samples)
            preds_matrix = np.array([r['preds'] for r in results])
            
            # 条件内での平均予測値 (Scale Factorを適用して損傷回数に変換)
            mean_preds_raw = np.mean(preds_matrix, axis=0)
            mean_preds_scaled = mean_preds_raw * self.SCALE_FACTOR
            
            # 条件内での平均CCCスコア
            mean_ccc = np.mean([r['val_ccc'] for r in results])
            
            # 負のCCCは重み計算でおかしくなるため0クリップなどを検討すべきだが、
            # 今回は上位条件前提なので正の値と仮定
            if mean_ccc < 0: mean_ccc = 0.0
            
            condition_summaries[eng_mode] = {
                'mean_preds': mean_preds_scaled,
                'mean_ccc': mean_ccc,
                'std_preds': np.std(preds_matrix * self.SCALE_FACTOR, axis=0) # ばらつき確認用
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
        for eng_mode, summary in condition_summaries.items():
            # 重みの計算: 自条件のCCC / 全条件のCCC合計
            weight = summary['mean_ccc'] / total_ccc_score
            weighted_preds = summary['mean_preds'] * weight
            final_preds += weighted_preds
            
            weights_info.append(f"{eng_mode}: Weight={weight:.3f} (CCC={summary['mean_ccc']:.3f})")
            
        print("Weights used:", ", ".join(weights_info))

        # ---------------------------------------------------------
        # 4. 信頼区間 (±2sigma) の算出
        # ---------------------------------------------------------
        # ここでは簡易的に、各条件の予測値の分散を合成するのではなく、
        # 条件ごとの標準偏差の「加重平均」を最終的な不確実性として採用します
        # (または、全予測分布の標準偏差を計算するアプローチもありますが、シンプルにします)
        final_std = np.zeros(n_samples)
        for eng_mode, summary in condition_summaries.items():
            weight = summary['mean_ccc'] / total_ccc_score
            final_std += summary['std_preds'] * weight
            
        lower_bound = final_preds - 2 * final_std
        upper_bound = final_preds + 2 * final_std
        
        # 負の値の補正
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
        
        # ファイル名から情報をパースしてソート
        try:
            # 想定形式: 0121_000V3F.csv -> 000V3F -> GearID, ToothID
            df_result['GearID'] = df_result['FileName'].apply(lambda x: x.split('_')[1][5:7])
            df_result['ToothID'] = df_result['FileName'].apply(lambda x: x.split('_')[1][8:10])
            df_result = df_result.sort_values(by=['GearID', 'ToothID'])
            df_result = df_result.drop(columns=['GearID', 'ToothID'])
        except:
            pass # パース失敗時はそのまま

        out_path = os.path.join(self.output_dir, "Final_Weighted_Ensemble_Result.csv")
        df_result.to_csv(out_path, index=False)
        print(f"\n✅ Final weighted ensemble results saved to: {out_path}")
        print(df_result.head())

# ==============================================================================
# 6. Main Execution Flow
# ==============================================================================
def main():
    print("🚀 Starting Weighted Ensemble Process...")
    
    config = ExperimentConfig()
    data_mgr = DataManager(config)
    trainer = ModelTrainer(config)
    aggregator = HybridEnsembleAggregator(config.output_dir)

    # 1. Load Data
    df_train_full = data_mgr.load_train_data()
    df_blind_full = data_mgr.load_blind_data()
    
    if df_train_full is None or df_blind_full is None:
        print("❌ Data loading failed.")
        return

    # 2. Iterate Conditions (Top conditions only)
    for model_type in config.model_types:
        for input_mode in config.input_modes:
            for use_rot in config.rotation_strategies:
                for eng_mode in config.eng_modes:
                    # ★ 変更点1: R面データのドロップループを追加
                    for drop_r in config.drop_r_data_options:
                        for (f_min, f_max, f_label) in config.freq_ranges:
                            
                            unique_id = f"{model_type}_{input_mode}_{'ROT' if use_rot else 'NOROT'}_{eng_mode}_{f_label}"
                            if drop_r: unique_id += "_NoR"
                            print(f"\n👉 Processing: {unique_id}")
                            
                            # フィルタリング
                            df_tr_curr = df_train_full[
                                (df_train_full['[Hz]'] >= f_min) & (df_train_full['[Hz]'] <= f_max)
                            ].copy()
                            df_bl_curr = df_blind_full[
                                (df_blind_full['[Hz]'] >= f_min) & (df_blind_full['[Hz]'] <= f_max)
                            ].copy()
                            
                            # ★ 変更点2: R面データの除外 (Training & Blind)
                            if drop_r:
                                print("   ℹ Filtering out 'R' side data...")
                                # sideカラムが存在することを前提にフィルタリング
                                if 'side' in df_tr_curr.columns:
                                    df_tr_curr = df_tr_curr[df_tr_curr['side'] == 'F'].copy()
                                if 'side' in df_bl_curr.columns:
                                    df_bl_curr = df_bl_curr[df_bl_curr['side'] == 'F'].copy()
                            
                            if df_tr_curr.empty or df_bl_curr.empty: 
                                print("   ⚠ Empty dataframe after filtering. Skipping.")
                                continue

                            # 複数回試行
                            for trial in range(config.n_trials):
                                print(f"   [Trial {trial+1}/{config.n_trials}]")
                                
                                # 前処理 & 特徴量生成
                                preproc = StrictPreprocessor(config, input_mode, use_rot)
                                
                                df_tr_trans = preproc.transform(df_tr_curr)
                                df_bl_trans = preproc.transform(df_bl_curr)
                                
                                df_tr_fin = preproc.apply_feature_engineering(df_tr_trans, eng_mode)
                                df_bl_fin = preproc.apply_feature_engineering(df_bl_trans, eng_mode)
                                
                                X_tr, y_tr, _, _ = preproc.generate_sequences(df_tr_fin, is_blind=False)
                                X_bl, _, bl_conds, _ = preproc.generate_sequences(df_bl_fin, is_blind=True)
                                
                                if len(X_tr) == 0 or len(X_bl) == 0: continue
                                
                                # 学習 & 予測
                                res = trainer.train_full_and_predict(X_tr, y_tr, X_bl, model_type)
                                
                                # 結果の蓄積
                                aggregator.add_trial_result(
                                    unique_id, 
                                    res['val_ccc'], 
                                    res['preds_inf'], 
                                    bl_conds
                                )

    # === 最終アンサンブル実行 (重み付き平均) ===
    aggregator.run_ensemble()
    print("✅ All Finished.")

if __name__ == "__main__":
    main()