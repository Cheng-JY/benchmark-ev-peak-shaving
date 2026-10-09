r"""Collect V2G optimizer summaries from a local data directory.

Usage (Windows or Linux):
  python ecv2g\user_dirs\jcheng\scripts\analysis\collect_v2g_scenarios.py --ev-count 73
  python ecv2g\user_dirs\jcheng\scripts\analysis\collect_v2g_scenarios.py --base-dir data/data_20260928 --scenarios-dir data/data_20260928_1/scenarios --scan-dir data/data_20260928_2 --output data/data_20260928/results.csv --ev-count 97
  python ecv2g\user_dirs\jcheng\scripts\analysis\collect_v2g_scenarios.py --base-dir data/data_20260928 --scenarios-dir data/data_20260928_1/scenarios --scan-dir data/data_20260928_2 --ev-count 73 --output results.csv

Relative folder arguments are resolved from the current working directory.
With no folder arguments, the script searches current and parent directories
for data/data_20260928/scenario_summary.csv.
Failed optimizations use their group's uncontrolled peak as optimized_peak_MW
while retaining status=failed and the original error message.
Runs that fail because required EV CSV columns are missing are omitted.
The parameter scan in data_20260928_2.zip is labeled separately; its EV count
comes from the optimizer summary rather than a user supplied assumption.
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


FIELDS = [
    "experiment", "net_name", "station_name", "region_name", "scenario_id",
    "household_ratio", "scenario_hp_ratio", "assigned_heat_pumps",
    "scenario_ev_ratio", "climate_model", "climate_year", "seed",
    "option", "ev_count", "controlled_ev_count", "uncontrolled_ev_count",
    "participation_rate_requested", "participation_rate_realized", "ch_refuel", "flexible_window",
    "charging_avail_reduction_type", "charging_avail_reduction_fraction",
    "uncontrolled_peak_MW", "optimized_peak_MW", "peak_threshold_MW",
    "peak_reduction_MW", "peak_reduction_percent", "status", "success", "error_type", "error_message",
    "availability_column", "soc_min_column", "soc_max_column", "charging_column",
    "start_discharging_kWh_per_vehicle", "charging_efficiency",
    "discharging_efficiency", "time_steps",
]


def reduced_availability(column):
    """Reduced availability.
    
    Parameters
    ----------
    column : object
        Column used by this operation.
    
    Returns
    -------
    object
        Result computed as ``(match.group(1), match.group(2)) if match else ('', '')``."""
    match = re.search(r"_reduced_(flat|peak)(\d+(?:\.\d+)?)_cp$", column or "")
    return (match.group(1), match.group(2)) if match else ("", "")


def make_row(meta, identifiers, result, archive_name, source, error=False):
    """Make row.
    
    Parameters
    ----------
    meta : object
        Meta used by this operation.
    identifiers : object
        Identifiers used by this operation.
    result : object
        Result used by this operation.
    archive_name : object
        Archive name used by this operation.
    source : object
        Source used by this operation.
    error : object
        Error used by this operation.
    
    Returns
    -------
    object
        Result computed as ``row``."""
    scenario_id, ev_folder, participation, ch_refuel = identifiers
    reduction_type, reduction = reduced_availability(
        result.get("availability_column", result.get("ch_avail_column", ""))
    )
    row = dict.fromkeys(FIELDS, "")
    row.update({
        "experiment": "ch_refuel_availability_scan" if ch_refuel else "ev_participation_hp_scan",
        "net_name": meta.get("net_name", ""), "station_name": meta.get("station_name", ""),
        "region_name": meta.get("region_name", ""), "scenario_id": scenario_id,
        "household_ratio": meta.get("household_ratio", ""),
        "scenario_hp_ratio": meta.get("scenario_hp_ratio", ""),
        "assigned_heat_pumps": meta.get("assigned_heat_pumps", ""),
        "scenario_ev_ratio": meta.get("scenario_ev_ratio", ""),
        "climate_model": meta.get("hp_timeseries_variant", ""),
        "climate_year": meta.get("climate_year", ""), "seed": meta.get("seed", ""),
        "option": result.get("option", ""),
        "ev_count": result.get("total_vehicle_count") or result.get("vehicle_count") or ev_folder,
        "controlled_ev_count": result.get("controlled_vehicle_count", ""),
        "uncontrolled_ev_count": result.get("uncontrolled_vehicle_count", ""),
        "participation_rate_requested": participation,
        "participation_rate_realized": result.get("n_controlled_realized", ""),
        "ch_refuel": "1.0" if result.get("option") == "1" else (ch_refuel or "0.4"),
        "charging_avail_reduction_type": reduction_type,
        "charging_avail_reduction_fraction": reduction,
        "status": result.get("run_status", "failed" if error else "success"),
        "error_type": result.get("error_type", ""),
        "error_message": result.get("error_message", ""),
        "availability_column": result.get("availability_column") or result.get("ch_avail_column", ""),
        "soc_min_column": result.get("soc_min_column", ""),
        "soc_max_column": result.get("soc_max_column", ""),
        "charging_column": result.get("charging_column", ""),
    })
    row["flexible_window"] = "0" if "soc_min_lim_refuel" in row["soc_min_column"].lower() else "1"
    row["success"] = "1" if row["status"].lower() == "success" else "0"
    if not error:
        for key in ("uncontrolled_peak_MW", "optimized_peak_MW", "peak_threshold_MW", "peak_reduction_MW",
                    "peak_reduction_percent", "start_discharging_kWh_per_vehicle",
                    "charging_efficiency", "discharging_efficiency", "time_steps"):
            row[key] = result.get(key, "")
    return row


def identify(name, scan):
    """Identify .
    
    Parameters
    ----------
    name : object
        Name used by this operation.
    scan : object
        Scan used by this operation.
    
    Returns
    -------
    object
        Computed result for the selected input and execution branch."""
    parts = name.split("/")
    if scan:
        match = next((re.fullmatch(r"greedy_results_(0\.\d+|1\.0)", part)
                      for part in parts if re.fullmatch(r"greedy_results_(0\.\d+|1\.0)", part)), None)
        if not match:
            return None
        return ("data_20260928_2", "", "1.0", match.group(1))
    scenario_index = next((i for i, p in enumerate(parts)
                           if re.fullmatch(r"hh_.+__hp_.+__ev_.+__.+__seed_\d+", p)), None)
    if scenario_index is None or len(parts) < scenario_index + 4:
        return None
    ev = re.fullmatch(r"data_\d+_(\d+)", parts[scenario_index + 1])
    rate = re.fullmatch(r"greedy_results_nctrl(\d+(?:\.\d+)?)", parts[scenario_index + 2])
    if not ev or not rate:
        return None
    return (parts[scenario_index], ev.group(1), rate.group(1), "")


def local_metadata(root):
    """Local metadata.
    
    Parameters
    ----------
    root : object
        Root used by this operation.
    
    Returns
    -------
    object
        Result computed as ``{r['scenario_id']: r for r in csv.DictReader(inp, delimiter=';')}``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    summaries = list(root.rglob("scenario_summary.csv"))
    if len(summaries) != 1:
        raise ValueError(f"Expected one scenario_summary.csv under {root}, got {len(summaries)}")
    with summaries[0].open(encoding="utf-8-sig", newline="") as inp:
        return {r["scenario_id"]: r for r in csv.DictReader(inp, delimiter=";")}


def metadata_for(scenario_id, metadata):
    """Metadata for.
    
    Parameters
    ----------
    scenario_id : object
        Scenario id used by this operation.
    metadata : object
        Metadata used by this operation.
    
    Returns
    -------
    object
        Computed result for the selected input and execution branch.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    if scenario_id in metadata:
        return metadata[scenario_id]
    # The zero-HP optimization can exist without a generated base-load
    # scenario. Reuse only invariant identifiers from the matching HP runs.
    match = re.fullmatch(r"hh_(.+)__hp_0__ev_(.+)__(.+)__seed_(\d+)", scenario_id)
    if not match:
        raise ValueError(f"No scenario metadata for {scenario_id}")
    hh, ev, climate, seed = match.groups()
    candidates = [r for key, r in metadata.items()
                  if re.fullmatch(rf"hh_{re.escape(hh)}__hp_[^_]+__ev_{re.escape(ev)}__{re.escape(climate)}__seed_{seed}", key)]
    if not candidates:
        raise ValueError(f"No matching HP scenario metadata for {scenario_id}")
    stable = ("net_name", "station_name", "region_name", "household_ratio",
              "scenario_ev_ratio", "hp_timeseries_variant", "climate_year", "seed",
              "household_count_generated", "household_count_base")
    for key in stable:
        if len({r.get(key, "") for r in candidates}) != 1:
            raise ValueError(f"Conflicting {key} among metadata candidates for {scenario_id}")
    inferred = {key: candidates[0].get(key, "") for key in stable}
    inferred.update(scenario_hp_ratio="0", assigned_heat_pumps="0")
    return inferred


def resolve_ev_folder(ev_root, ev_count):
    """Find a fleet input folder independently of optimization outputs."""
    count = str(int(float(ev_count)))
    date = re.match(r'data_(\d+)', ev_root.name)
    if date:
        expected = ev_root / f'data_{date.group(1)}_{count}'
        if expected.is_dir():
            return expected
    matches = [p for p in ev_root.rglob('data_*_' + count)
               if p.is_dir() and re.fullmatch(r'data_\d+_' + re.escape(count), p.name)]
    if len(matches) == 1:
        return matches[0]
    if re.fullmatch(r'data_\d+_' + re.escape(count), ev_root.name):
        return ev_root
    if not matches:
        raise ValueError(f'No EV input directory for {count} vehicles under {ev_root}')
    raise ValueError(f'Multiple EV input directories for {count} vehicles under {ev_root}; narrow --ev-dir')



def select_ev_profile(candidates, actual_refuel, baseline_file='', scan=False):
    """Resolve a profile without confusing the option-1 label with its input."""
    if not scan and baseline_file:
        basename = str(baseline_file).replace('\\', '/').split('/')[-1]
        exact = [p for p in candidates if p.name == basename]
        if len(exact) == 1:
            return exact
    # The full-participation EV baseline is the fleet input requested here.
    full = [p for p in candidates if re.search(
        r'(?:^|[_-])n(?:ctrl|ctl)1(?:\.0+)?(?=[_.-]|$)', p.name, re.IGNORECASE)]
    if full:
        return full
    refuel = float(actual_refuel)
    # Prefer canonical fleet profiles over separately generated reduced files.
    canonical = []
    for path in candidates:
        match = re.search(r'_R\d+_(\d+(?:\.\d+)?)\.csv$', path.name, re.IGNORECASE)
        if match and abs(float(match.group(1)) - refuel) < 1e-8:
            canonical.append(path)
    if canonical:
        return canonical
    suffix = []
    for path in candidates:
        match = re.search(r'_(\d+(?:\.\d+)?)\.csv$', path.name)
        if match and abs(float(match.group(1))-refuel) < 1e-8 and '_reduced_' not in path.name:
            suffix.append(path)
    return suffix if suffix else (candidates if len(candidates) == 1 else [])


def collect_folders(locations, metadata, scan_ev_count, ev_root=None):
    """Collect folders.
    
    Parameters
    ----------
    locations : object
        Locations used by this operation.
    metadata : object
        Metadata used by this operation.
    scan_ev_count : object
        Scan ev count used by this operation.
    ev_root : object
        Ev root used by this operation.
    
    Returns
    -------
    object
        Result computed as ``list(found.values())``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    found = {}
    baseline = {}
    templates = {}
    scan_metadata_candidates = [r for k, r in metadata.items()
                                if re.fullmatch(r"hh_1__hp_1__ev_1__.+__seed_\d+", k)]
    for label, folder in locations:
        if not folder.is_dir():
            raise ValueError(f"{label} is not a directory: {folder}")
        for path in folder.rglob("*.csv"):
            if path.name not in ("greedy_all_options_summary.csv", "greedy_failed_options.csv"):
                continue
            source = path.relative_to(folder).as_posix()
            scan = label == "scan"
            ids = identify(source, scan)
            if not ids:
                continue
            if scan:
                meta = scan_metadata_candidates[0] if len(scan_metadata_candidates) == 1 else {}
            else:
                meta = metadata_for(ids[0], metadata)
            error = path.name == "greedy_failed_options.csv"
            with path.open(encoding="utf-8-sig", newline="") as inp:
                for result in csv.DictReader(inp, delimiter=";"):
                    row = make_row(meta, ids, result, "", f"{label}/{source}", error)
                    if row["status"].lower() == "failed" and row["error_message"].strip().startswith(
                        "Required EV CSV column(s) are missing:"
                    ):
                        continue
                    if scan:
                        row["ev_count"] = str(scan_ev_count)
                        row["controlled_ev_count"] = str(scan_ev_count)
                        row["uncontrolled_ev_count"] = "0"
                        row["participation_rate_realized"] = "1.0"
                        row["household_ratio"] = "1.0"
                        row["scenario_hp_ratio"] = "1.0"
                    reduction_type, reduction = reduced_availability(row["availability_column"])
                    row["charging_avail_reduction_type"] = reduction_type or "flat"
                    row["charging_avail_reduction_fraction"] = reduction or "1.0"
                    if not error and row["uncontrolled_peak_MW"]:
                        group = ids
                        previous = baseline.get(group)
                        if previous is not None and abs(float(previous) - float(row["uncontrolled_peak_MW"])) > 1e-5:
                            raise ValueError(f"Inconsistent uncontrolled peaks in {group}: {previous}, {row['uncontrolled_peak_MW']}")
                        baseline[group] = row["uncontrolled_peak_MW"]
                        templates.setdefault(group, row)
                    key = (ids, row["option"], row["charging_avail_reduction_type"],
                           row["charging_avail_reduction_fraction"])
                    if key in found:
                        raise ValueError(f"Duplicate result: {path}, {key}")
                    row['_feature_context'] = {'result_dir': str(path.parent), 'meta': meta, 'scan': scan,
                                               'ch_refuel_actual': ids[3] or '0.4',
                                               'baseline_ev_file': result.get('uncontrolled_baseline_ev_csv', ''),
                                               'ev_folder': str(folder if scan else path.parent.parent),
                                               'ev_root': str(ev_root) if ev_root is not None else ''}
                    found[key] = row
    for key, row in found.items():
        if row["status"] == "failed":
            value = baseline.get(key[0])
            if value is None:
                raise ValueError(f"Cannot find uncontrolled baseline for failed result: {key}")
            template = templates[key[0]]
            for field in ("controlled_ev_count", "uncontrolled_ev_count", "participation_rate_realized",
                          "charging_efficiency", "discharging_efficiency", "time_steps"):
                if not row[field]:
                    row[field] = template[field]
            row["uncontrolled_peak_MW"] = value
            row["optimized_peak_MW"] = value
            row["peak_threshold_MW"] = value
            row["peak_reduction_MW"] = "0"
            row["peak_reduction_percent"] = "0"
    return list(found.values())


PREDICTION_FIELDS = [
    'household_count', 'base_peak_MW', 'base_mean_MW', 'base_p95_MW',
    'base_load_factor', 'base_peak_hour', 'hp_share_at_base_peak',
    'ev_per_household', 'ev_power_mean_MW', 'ev_power_at_base_peak_MW',
    'soc_range_mean_MWh', 'soc_range_at_base_peak_MWh',
    'ev_power_to_base_peak', 'soc_range_to_base_peak_hours',
    'availability_family', 'soc_min_family', 'feature_calendar_alignment', 'feature_status', 'feature_message',
]


def prediction_features(rows, metadata, base_dir):
    """Prediction features.
    
    Parameters
    ----------
    rows : object
        Records processed by this operation.
    metadata : object
        Metadata used by this operation.
    base_dir : object
        Base dir used by this operation.
    
    Returns
    -------
    object
        Result computed as ``rows``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    import pandas as pd
    import numpy as np
    cache = {}

    def read(path):
        """Read .
        
        Parameters
        ----------
        path : object
            File or directory path used by this operation.
        
        Returns
        -------
        object
            Result computed as ``cache[key]``."""
        key = str(path)
        if key not in cache:
            cache[key] = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig')
        return cache[key]

    def hourly(path, column, scale=1.0):
        """Hourly .
        
        Parameters
        ----------
        path : object
            File or directory path used by this operation.
        column : object
            Column used by this operation.
        scale : object
            Scale used by this operation.
        
        Returns
        -------
        object
            Result computed as ``out``.
        
        Raises
        ------
        ValueError
            If an explicit validation or processing check in this operation fails."""
        df = read(path)
        index = pd.to_datetime(df.iloc[:, 0], errors='raise')
        values = pd.to_numeric(df[column], errors='raise').to_numpy() * scale
        out = pd.Series(values, index=index)
        if out.index.duplicated().any():
            raise ValueError(f'Duplicate timestamps in {path}')
        out = out.resample('h').mean()
        if out.isna().any():
            raise ValueError(f'Missing hourly input values in {path}')
        return out

    for row in rows:
        row.update(dict.fromkeys(PREDICTION_FIELDS, ''))
        context = row.pop('_feature_context', {})
        result_dir = Path(context.get('result_dir', '.'))
        for src, dst in [('availability_column', 'availability_family'), ('soc_min_column', 'soc_min_family')]:
            row[dst] = re.sub(r'_reduced_(?:flat|peak)\d+(?:\.\d+)?', '', row[src])
        try:
            meta = metadata_for(row['scenario_id'], metadata) if row['experiment'] != 'ch_refuel_availability_scan' else context.get('meta', {})
            hh = meta.get('household_count_generated') or meta.get('household_count_base')
            if hh:
                row['household_count'] = float(hh)
                if float(hh) > 0:
                    row['ev_per_household'] = float(row['ev_count']) / float(hh)
            # Read only pre-optimization inputs from the hourly export.
            own_export = result_dir / f"greedy_results_{row['option']}" / 'greedy_peak_shaving_results.csv'
            siblings = sorted(result_dir.glob('greedy_results_*/greedy_peak_shaving_results.csv'))
            base_export = own_export if own_export.is_file() else (siblings[0] if siblings else None)
            if base_export:
                base = hourly(base_export, 'base_load_MW')
                hp = hourly(base_export, 'heat_pump_load_MW')
            else:
                sid = meta.get('scenario_id', row['scenario_id'])
                scenario_folder = base_dir / sid
                # A missing zero-HP folder may reuse the matching household input.
                if not scenario_folder.is_dir() and float(row['scenario_hp_ratio']) == 0:
                    candidates = [base_dir / key for key in metadata if
                                  key.startswith('hh_' + str(row['household_ratio']).replace('.', 'p') + '__')]
                    candidates = [p for p in base_dir.glob('hh_*__hp_*__ev_*__*__seed_' + str(row['seed'])) if p.is_dir()]
                    if len(candidates) == 0:
                        raise ValueError('No household input for zero-HP scenario')
                    scenario_folder = candidates[0]
                base = hourly(scenario_folder / 'household' / 'household_load.csv', 'hh_load (MW)')
                if float(row['scenario_hp_ratio']) == 0:
                    hp = base * 0
                else:
                    hp = hourly(scenario_folder / 'heat_pump' / 'heat_pump_profiles.csv', 'total_all', .001)
                if not base.index.equals(hp.index):
                    raise ValueError('Household and heat-pump timestamps differ')
                base = base + hp
            peak = float(base.max())
            t = base.idxmax()
            row.update(base_peak_MW=peak, base_mean_MW=float(base.mean()),
                       base_p95_MW=float(base.quantile(.95)), base_peak_hour=t.hour,
                       base_load_factor=float(base.mean()/peak) if peak > 0 else '',
                       hp_share_at_base_peak=float(hp.loc[t]/peak) if peak > 0 else '')
            if own_export.is_file():
                power = hourly(own_export, 'charging_power_available_MW')
                lower = hourly(own_export, 'soc_min_MWh')
                upper = hourly(own_export, 'soc_max_MWh')
            else:
                ev_folder = Path(context.get('ev_folder', '.'))
                if not context.get('scan') and context.get('ev_root'):
                    ev_folder = resolve_ev_folder(Path(context['ev_root']), row['ev_count'])
                candidates = sorted(ev_folder.glob('ch_profiles*.csv'))
                # Failed summary rows may omit the input filename; recover it
                # from a successful sibling in the same EV/participation group.
                baseline_file = context.get('baseline_ev_file', '')
                # Prefer the EV baseline recorded for full participation.
                if not context.get('scan'):
                    for folder_name in ('greedy_results_nctrl1.0', 'greedy_results_nctl1.0',
                                        'greedy_results_nctrl1', 'greedy_results_nctl1'):
                        full_summary = result_dir.parent / folder_name / 'greedy_all_options_summary.csv'
                        if not full_summary.is_file():
                            continue
                        full_df = read(full_summary)
                        if 'uncontrolled_baseline_ev_csv' in full_df:
                            names = full_df['uncontrolled_baseline_ev_csv'].dropna().astype(str)
                            names = names[names.str.strip().ne('')].unique()
                            if len(names) == 1:
                                baseline_file = names[0]
                                break
                if not baseline_file and not context.get('scan'):
                    summary = read(result_dir / 'greedy_all_options_summary.csv')
                    if 'uncontrolled_baseline_ev_csv' in summary:
                        filenames = summary['uncontrolled_baseline_ev_csv'].dropna().astype(str).unique()
                        if len(filenames) == 1:
                            baseline_file = filenames[0]
                candidates = select_ev_profile(candidates, context['ch_refuel_actual'],
                                               baseline_file, context.get('scan', False))
                if len(candidates) != 1:
                    raise ValueError(f'Expected one matching EV input, found {len(candidates)}')
                if row['controlled_ev_count'] != '':
                    scale = float(row['controlled_ev_count']) / 1000.0
                else:
                    # Recover the exact fleet scaling from an exported input
                    # rather than guessing how participation was rounded.
                    scale = None
                    reference_summary = read(result_dir / 'greedy_all_options_summary.csv')
                    for _, reference in reference_summary.iterrows():
                        reference_path = result_dir / f"greedy_results_{int(reference['option'])}" / 'greedy_peak_shaving_results.csv'
                        column = reference.get('availability_column', '')
                        if not reference_path.is_file() or column not in read(candidates[0]):
                            continue
                        raw_power = hourly(candidates[0], column).to_numpy()
                        exported_power = hourly(reference_path, 'charging_power_available_MW').to_numpy()
                        mask = raw_power > 1e-9
                        if len(raw_power) != len(exported_power) or not mask.any():
                            continue
                        candidate_scale = float(np.median(exported_power[mask] / raw_power[mask]))
                        if np.allclose(raw_power * candidate_scale, exported_power, rtol=1e-6, atol=1e-9):
                            scale = candidate_scale
                            break
                    if scale is None:
                        raise ValueError('Controlled fleet scaling cannot be inferred from input exports')
                power = hourly(candidates[0], row['availability_column'], scale)
                lower = hourly(candidates[0], row['soc_min_column'], scale)
                upper = hourly(candidates[0], row['soc_max_column'], scale)
            if not base.index.equals(power.index):
                # Different year labels are accepted only when an existing run
                # proves the optimizer used the same EV input order and scale.
                verified = False
                summary = read(result_dir / 'greedy_all_options_summary.csv')
                for _, reference in summary.iterrows():
                    reference_path = result_dir / f"greedy_results_{int(reference['option'])}" / 'greedy_peak_shaving_results.csv'
                    column = reference.get('availability_column', '')
                    if not reference_path.is_file() or column not in read(candidates[0]):
                        continue
                    source_power = hourly(candidates[0], column, scale)
                    exported_power = hourly(reference_path, 'charging_power_available_MW')
                    if len(source_power) == len(exported_power) == len(base) and np.allclose(
                            source_power.to_numpy(), exported_power.to_numpy(), rtol=1e-6, atol=1e-9):
                        verified = True
                        break
                if not verified or any(len(v) != len(base) for v in (power, lower, upper)):
                    raise ValueError('EV calendar differs and positional alignment could not be verified')
                power.index = lower.index = upper.index = base.index
                row['feature_calendar_alignment'] = 'position_verified_against_export'
            else:
                row['feature_calendar_alignment'] = 'timestamps'
            if not base.index.equals(lower.index) or not base.index.equals(upper.index):
                raise ValueError('SoC and base-load timestamps differ')
            width = upper - lower
            if (width < -1e-8).any():
                raise ValueError('soc_max is smaller than soc_min in EV inputs')
            row.update(ev_power_mean_MW=float(power.mean()), ev_power_at_base_peak_MW=float(power.loc[t]),
                       soc_range_mean_MWh=float(width.mean()), soc_range_at_base_peak_MWh=float(width.loc[t]),
                       ev_power_to_base_peak=float(power.loc[t]/peak) if peak > 0 else '',
                       soc_range_to_base_peak_hours=float(width.loc[t]/peak) if peak > 0 else '',
                       feature_status='complete')
        except (ValueError, KeyError, OSError, TypeError) as exc:
            row['feature_status'] = 'partial'
            row['feature_message'] = str(exc)
    return rows


def main():
    """Run the command-line workflow for this module."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, help="Parent data folder containing the three standard subfolders")
    parser.add_argument("--base-dir", type=Path, help="Folder containing scenario_summary.csv")
    parser.add_argument("--scenarios-dir", type=Path, help="Folder containing EV/participation/HP optimization results")
    parser.add_argument("--scan-dir", type=Path, help="Folder containing ch_refuel and availability scan results")
    parser.add_argument("--ev-count", type=int, help="EV count to assign to every result in --scan-dir")
    parser.add_argument("--ev-dir", type=Path, help="EV input root, e.g. data/data_20260928_1 containing data_20260928_N")
    parser.add_argument("--prediction-features", action="store_true", help="Extract hourly pre-optimization base-load and EV descriptors")
    parser.add_argument("--exclude-options", type=int, nargs="*", default=[3], help="Default excludes option 3; pass flag alone to keep all")
    parser.add_argument("--output", type=Path, default=Path("v2g_scenario_summary.csv"))
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    working_dir = Path.cwd()

    def resolve(path):
        """Resolve .
        
        Parameters
        ----------
        path : object
            File or directory path used by this operation.
        
        Returns
        -------
        object
            Result computed as ``path if path is None or path.is_absolute() else working_dir / path``."""
        return path if path is None or path.is_absolute() else working_dir / path

    args.root = resolve(args.root)
    args.base_dir = resolve(args.base_dir)
    args.scenarios_dir = resolve(args.scenarios_dir)
    args.scan_dir = resolve(args.scan_dir)
    args.output = resolve(args.output)
    args.ev_dir = resolve(args.ev_dir)
    if args.root is None and all(x is None for x in
                                 (args.base_dir, args.scenarios_dir, args.scan_dir)):
        candidates = list(dict.fromkeys(
            parent / "data" for start in (working_dir, script_dir) for parent in (start, *start.parents)
        ))
        args.root = next((folder for folder in candidates
                          if (folder / "data_20260928" / "scenario_summary.csv").is_file()), None)
        if args.root is None:
            parser.error("Cannot locate data/data_20260928/scenario_summary.csv. "
                         "Pass --root PATH_TO_DATA or the three folder arguments.")
    if args.root:
        if any((args.base_dir, args.scenarios_dir, args.scan_dir)):
            parser.error("Use --root or the separate folder arguments, not both")
        base_dir = args.root / "data_20260928"
        scenarios_dir = args.root / "data_20260928_1"
        scan_dir = args.root / "data_20260928_2"
    else:
        base_dir, scenarios_dir, scan_dir = args.base_dir, args.scenarios_dir, args.scan_dir
        if base_dir is None:
            parser.error("Specify --root or --base-dir")
        if scenarios_dir is None and scan_dir is None:
            parser.error("Specify at least --scenarios-dir or --scan-dir")
    locations = [(label, folder) for label, folder in
                 (("scenarios", scenarios_dir), ("scan", scan_dir)) if folder is not None and folder.is_dir()]
    if not locations:
        parser.error(f"No scenario or scan directory found. Checked: {scenarios_dir}, {scan_dir}. "
                     "Pass --scenarios-dir and/or --scan-dir with your actual paths.")
    if any(label == "scan" for label, _ in locations) and (args.ev_count is None or args.ev_count < 0):
        parser.error("Provide --ev-count N for the data_20260928_2 scan results")
    for label, folder in (("scenarios", scenarios_dir), ("scan", scan_dir)):
        if folder is not None and not folder.is_dir() and not args.root:
            parser.error(f"--{label}-dir does not exist: {folder}")
    ev_root = args.ev_dir
    if ev_root is None and scenarios_dir is not None:
        ev_root = scenarios_dir.parent if scenarios_dir.name.lower() in ('scenario', 'scenarios') else scenarios_dir
    rows = collect_folders(locations, local_metadata(base_dir), args.ev_count, ev_root)
    rows = [r for r in rows if int(r['option']) not in args.exclude_options]
    # Exclude this exact availability / minimum-SoC combination.
    rows = [r for r in rows if not (
        r.get('availability_column', '').strip() == 'ch_avail_refuel_targetsoc_cp'
        and r.get('soc_min_column', '').strip() == 'soc_min_refuel_cp'
    )]
    if args.prediction_features:
        rows = prediction_features(rows, local_metadata(base_dir), base_dir)
    else:
        for row in rows:
            row.pop('_feature_context', None)
    output_fields = FIELDS + (PREDICTION_FIELDS if args.prediction_features else [])
    # Option 1 ignores ch_refuel; after labeling it 1.0, identical scan runs
    # should contribute one observation rather than ten copies.
    rows = list({tuple(row[key] for key in output_fields): row for row in rows}.values())
    rows.sort(key=lambda r: (r["experiment"], r["scenario_hp_ratio"], r["ch_refuel"],
                             int(r["ev_count"] or 0), float(r["participation_rate_requested"] or 0),
                             int(r["option"] or 0)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, output_fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.output}")
    for (experiment, status), count in sorted(Counter((r["experiment"], r["status"]) for r in rows).items()):
        print(f"  {experiment}: {status} = {count}")


if __name__ == "__main__":
    main()
