"""Ev assignment builder utilities for the EV/V2G analysis workflow."""

# components/input_data/vehicle_data/ev_assignment_builder

import pandas as pd
import numpy as np

from ....utils._calculation import calculate_plug_in_duration

class EVAssignmentBuilder:

    """Represent evassignment builder and its associated operations."""
    def __init__(self, scenario_data, scenario_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        scenario_data : object
            Scenario data used by this operation.
        scenario_config : object
            Scenario config used by this operation."""
        self.scenario_data = scenario_data
        self.cfg = scenario_config

        self.ev_dir = scenario_data.ev_dir

        self.dest_map = {1:"work", 2:"other", 3:"home", 4:"travel", 5:"highway"}

    # Plug-in Duration
    def compute_plug_in_duration(
            self,
            dest_name="home",
            des=3,
            ch_refuel_list=None):
        """Compute plug in duration.
        
        Parameters
        ----------
        dest_name : object
            Dest name used by this operation.
        des : object
            Des used by this operation.
        ch_refuel_list : object
            Ch refuel list used by this operation."""
        print("Calculating plug-in duration:")

        if ch_refuel_list is None:
            print("ch_refuel_list is None.")
            return
        
        date_input = self.cfg.date_input
        ev_results = {}

        for ch_refuel in ch_refuel_list:
            ch_refeul_use = 0.9 if ch_refuel == 1 else ch_refuel
            file_name = f"ch_profiles_{ch_refeul_use}_DEU_{date_input}_2012_cal2012_{self.cfg.region_name}_part00.parquet"
            file_path = self.ev_dir / file_name

            if not file_path.exists():
                print(f"⚠ Warning: {str(file_path)} not found, skipping.")

            charging_profiles = pd.read_parquet(file_path)
            keys = charging_profiles.columns.levels[0]
            
            for key in keys:
                profile = charging_profiles[key]

                plug_state = "_plug2" if ch_refuel != 1 else ""

            
                if key not in ev_results:
                    ev_results[key] = {"key": key}
                
                month_result = []
                for month in range(1, 13):
                    duration = calculate_plug_in_duration(profile, des, month, plug_state)
                    field_name = f"{dest_name}_{month}_{ch_refuel}"
                    ev_results[key][field_name] = duration
                    month_result.append(duration)

                ev_results[key][f"{dest_name}_max_{ch_refuel}"] = np.max(month_result)
                ev_results[key][f"{dest_name}_min_{ch_refuel}"] = np.min(month_result)
                ev_results[key][f"{dest_name}_avg_{ch_refuel}"] = np.mean(month_result)

        df_result = pd.DataFrame.from_dict(ev_results, orient='index')
        plug_path = self.ev_dir / "plug_in_duration.csv"
        df_result.to_csv(plug_path, sep=";", index=False)

    
    def assign_ev_to_household(self, pattern_list=None):
        """Assign ev to household.
        
        Parameters
        ----------
        pattern_list : object
            Pattern list used by this operation."""
        if pattern_list is None:
            pattern_list = ['..-.....-....-..[^5]...-H..A00S000-..']
        
        region_group = self.cfg.region_name
        net_pop_data_path = self.scenario_data.folder / "net_pop_data.csv"
        plug_in_duration_path = self.ev_dir / "plug_in_duration.csv"

        net_pop_data = pd.read_csv(net_pop_data_path, sep=";", low_memory=False)
        plug_in_duration_df = pd.read_csv(plug_in_duration_path, sep=';')
        ev_id_list = plug_in_duration_df["key"].tolist()

        needed_cols = ["hh_id", "HHTyp", "EinkTyp", "total_ev_count", "cp_single_ev_count", "cp_double_ev_count"]
        for idx, c in enumerate(needed_cols):
            if c not in plug_in_duration_df.columns:
                plug_in_duration_df.insert(idx+1, c, pd.NA)
        
        for _,row in net_pop_data.iterrows():
            total_ev_count = int(row['total_ev_count'])
            if total_ev_count == 0:
                continue

            hh_id = row['hh_id']
            hhtype = row['HHtyp']
            einktype = row['Einktyp']
            cp_single_ev_count = row['cp_single_ev_count']
            cp_double_ev_count = row['cp_double_ev_count']

            modify_id_patterns = self._modify_id_pattern(region_group, hhtype, einktype, pattern_list)
            filtered_profile_list = []
            for _ in modify_id_patterns:
                filtered_profile_list += [col for col in ev_id_list if col[11]==str(hhtype[-1]) and col[12]==str(einktype[-1])]
            
            unassigned_mask = plug_in_duration_df['hh_id'].isna()
            unassigned_ids = [ev for ev in filtered_profile_list if ev in plug_in_duration_df.loc[unassigned_mask, 'key'].values]

            selected_ids = unassigned_ids[:total_ev_count]

            for ev_id in selected_ids:
                plug_in_duration_df.loc[plug_in_duration_df['key'] == ev_id, 'hh_id'] = hh_id
                plug_in_duration_df.loc[plug_in_duration_df['key'] == ev_id, 'HHTyp'] = hhtype
                plug_in_duration_df.loc[plug_in_duration_df['key'] == ev_id, 'EinkTyp'] = einktype
                plug_in_duration_df.loc[plug_in_duration_df['key'] == ev_id, 'total_ev_count'] = total_ev_count
                plug_in_duration_df.loc[plug_in_duration_df['key'] == ev_id, 'cp_single_ev_count'] = cp_single_ev_count
                plug_in_duration_df.loc[plug_in_duration_df['key'] == ev_id, 'cp_double_ev_count'] = cp_double_ev_count

        plug_in_duration_df = plug_in_duration_df.dropna(subset=['hh_id'])

        output_file = self.ev_dir / "plug_in_duration_with_hh.csv"
        plug_in_duration_df.to_csv(output_file, sep=';', index=False)
                

    def _modify_id_pattern(self, region_group, hh, eink, patterns):
        """Modify id pattern.
        
        Parameters
        ----------
        region_group : object
            Region group used by this operation.
        hh : object
            Hh used by this operation.
        eink : object
            Eink used by this operation.
        patterns : object
            Patterns used by this operation.
        
        Returns
        -------
        object
            Result computed as ``[p.replace('-....-', f'-.{region_num}{hh_num}{eink_num}-') for p in patterns]``."""
        region_num = str(region_group)[-1]
        hh_num = str(hh)[-1]
        eink_num = str(eink)[-1]
        return [p.replace("-....-", f'-.{region_num}{hh_num}{eink_num}-') for p in patterns]
    

    def compute_driving_overview(self):
        """Compute driving overview."""
        profiles_overview_path = self.ev_dir / "profiles_overview.csv"
        plug_in_duration_hh_path = self.ev_dir / "plug_in_duration_with_hh.csv"

        profiles_overview = pd.read_csv(profiles_overview_path, index_col=False)
        plug_in_duration_hh = pd.read_csv(plug_in_duration_hh_path, sep=";", index_col=False)

        # Filter rows in profiles_overview whose key exists in plug_in_duration
        profiles_overview["key_prefix"] = profiles_overview["key"].astype(str).str[:20]
        plug_in_duration_hh["key_prefix"] = plug_in_duration_hh["key"].astype(str).str[:20]

        target_prefixes = plug_in_duration_hh["key_prefix"].unique()

        # 按前 20 位匹配
        filtered_profiles = profiles_overview[
            profiles_overview["key_prefix"].isin(target_prefixes)
        ].copy()
    
        if "hh_id" in plug_in_duration_hh.columns:
            hh_mapping = plug_in_duration_hh[["key_prefix", "hh_id"]].drop_duplicates()
            filtered_profiles = pd.merge(filtered_profiles, hh_mapping, on="key_prefix", how="left")

        filtered_profiles = filtered_profiles.loc[:, ~filtered_profiles.columns.str.contains("^Unnamed")]
    

        if "vehicle_id" in filtered_profiles.columns:
            filtered_profiles = filtered_profiles.drop(columns=["vehicle_id"])
        
        filtered_profiles = filtered_profiles.drop_duplicates(subset=["key_prefix"])

        # Save the result
        save_path = self.ev_dir / "profiles_overview_filtered.csv"
        filtered_profiles.to_csv(save_path, sep=";", index=False)
    
    def analyze_duplicate_prefixes(self):
        # 1. 读取文件
        # 注意：根据你之前的文件格式，分隔符是分号 ';'
        """Analyze duplicate prefixes."""
        profiles_overview_path = self.ev_dir / "profiles_overview_filtered.csv"
        plug_in_duration_hh_path = self.ev_dir / "plug_in_duration_with_hh.csv"

        profiles_overview = pd.read_csv(profiles_overview_path, index_col=False)
        plug_in_duration_hh = pd.read_csv(plug_in_duration_hh_path, sep=";", index_col=False)
        try:
            df_filtered = pd.read_csv(profiles_overview_path, sep=';')
        except FileNotFoundError:
            print("未找到 profiles_overview_filtered.csv 文件。请确保文件已上传或生成。")
            return

        # 2. 检查是否存在 key_prefix 列，如果没有则从 key 生成
        if 'key_prefix' not in df_filtered.columns:
            if 'key' in df_filtered.columns:
                df_filtered['key_prefix'] = df_filtered['key'].astype(str).str[:20]
            else:
                print("错误：文件中没有 'key' 或 'key_prefix' 列。")
                return

        # 3. 统计每个前缀出现的次数
        prefix_counts = df_filtered['key_prefix'].value_counts()

        # 4. 筛选出出现次数大于 1 的前缀 (即“一对多”的情况)
        duplicates = prefix_counts[prefix_counts > 1]

        # 5. 输出结果
        print(f"总行数: {len(df_filtered)}")
        print(f"唯一前缀数量: {len(prefix_counts)}")
        print(f"存在重复的前缀数量: {len(duplicates)}")
        print("-" * 30)
        
        if len(duplicates) > 0:
            print("以下前缀对应了多辆车 (前缀: 车辆数量):")
            print(duplicates)
            
            # 也可以打印出具体的 key 来看看
            print("\n具体示例 (展示前3个重复组):")
            for prefix in duplicates.index[:3]:
                print(f"\n前缀: {prefix}")
                matching_rows = df_filtered[df_filtered['key_prefix'] == prefix]['key']
                for key in matching_rows:
                    print(f"  - {key}")
        else:
            print("没有发现重复前缀。每行都是唯一的 (一对一关系)。")
