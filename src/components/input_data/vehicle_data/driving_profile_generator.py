"""Driving profile generator utilities for the EV/V2G analysis workflow."""

from __future__ import annotations

import pandas as pd

from ekfzladeprofile.main import main as run_ekfzladeprofile
import ekfzladeprofile.ladeprofile.a3_Ladeverfuegbarkeit as charging_availability

from ....utils._generate_profile import get_veh_per_mob
from ....utils._sample import get_ev_statistics


def assign_home_charging_type(
    keys,
    loading_type_input: pd.DataFrame,
    profiles_df: pd.DataFrame,
    mob_group: str | None = None,
    random_seed: int | None = None,
) -> list[str]:
    """Force every generated EV profile to use home charging.

    The additional arguments are kept because current ekfzladeprofile calls
    ``assign_ch_type`` with this complete function signature.
    """
    del loading_type_input, profiles_df, mob_group, random_seed
    return ["Home"] * len(keys)


# Preserve the existing ecv2g assumption that every selected EV has a home
# charging point. Only the charging-type function is replaced; the package
# path and the ekfzladeprofile input files remain unchanged.
charging_availability.assign_ch_type = assign_home_charging_type


class DrivingProfileGenerator:
    """Represent driving profile generator and its associated operations."""
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
        self.mob_group_list: list[int] = []

    def generate_profiles(
        self,
        net_pop_data: pd.DataFrame,
        filters: dict[str, list[str]],
    ) -> None:
        """Generate profiles.
        
        Parameters
        ----------
        net_pop_data : pd.DataFrame
            Net pop data used by this operation.
        filters : dict[str, list[str]]
            Filters used by this operation.
        
        Returns
        -------
        None"""
        cfg = self.cfg

        # An absolute path is accepted by the current ekfzladeprofile
        # data_path() implementation. Therefore, no ekfz_path or data_path
        # monkey-patching is required.
        data_folder_name = str(self.ev_dir.resolve())

        net_pop_data_copy, grouped = get_ev_statistics(net_pop_data)
        ev_stats = get_veh_per_mob(grouped, cfg.region_name)

        net_pop_data_path = self.scenario_data.folder / "net_pop_data.csv"
        net_pop_data_copy.to_csv(net_pop_data_path, sep=";", index=False)

        # Avoid duplicate entries when the same generator instance is reused.
        self.mob_group_list.clear()

        # The current ekfzladeprofile main() reads Kalenderjahr and Wetterjahr
        # from its scenario-definition sheet. ecv2g overrides only the
        # scenario year used for vehicle and charging parameters.
        scenario = {
            "Szenariojahr": int(cfg.scenario_year),
        }

        for _, row in ev_stats.iterrows():
            mob_group = int(row["mob_group"])
            veh_per_region = int(row["veh_per_region"])

            if veh_per_region <= 0:
                continue

            self.mob_group_list.append(mob_group)

            run_ekfzladeprofile(
                data_folder_name=data_folder_name,
                filters=filters,
                date_input=cfg.date_input,
                split_size=500,
                mob_groups=[mob_group],
                veh_per_region=veh_per_region,
                parallel=True,
                max_workers=7,
                random_seed=cfg.seed,
                scenario=scenario,
                up_to="generate_charging_availibility_profiles",

                # main() now removes intermediate outputs by default. These
                # files are still required by MixProfileBuilder, so this stage
                # must preserve them.
                cleanup_outputs=False,
            )

    