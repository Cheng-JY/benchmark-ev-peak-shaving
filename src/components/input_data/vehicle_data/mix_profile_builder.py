"""Mix profile builder utilities for the EV/V2G analysis workflow."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from joblib import parallel_backend

from ekfzladeprofile.ladeprofile.a1_Database import data_path as ekfz_data_path
from ekfzladeprofile.ladeprofile.a4_profile_mixer import mix_profiles
import ekfzladeprofile.ladeprofile.a4_SOC_Generation as soc_mod


@contextmanager
def temporary_ch_refuel_threshold(threshold: float) -> Iterator[None]:
    """Temporarily override the ekfzladeprofile refuelling threshold.

    The current ekfzladeprofile ``mix_profiles`` interface does not expose
    ``ch_refuel_threshold`` as a function argument. The value is read as the
    module-level constant ``CH_REFUEL_THRESHOLD`` in ``a4_SOC_Generation``.
    This context manager changes it only for one mix run and restores the
    original Excel-configured value afterwards.
    """
    original_threshold = soc_mod.CH_REFUEL_THRESHOLD
    soc_mod.CH_REFUEL_THRESHOLD = float(threshold)
    try:
        yield
    finally:
        soc_mod.CH_REFUEL_THRESHOLD = original_threshold


class MixProfileBuilder:
    """Represent mix profile builder and its associated operations."""
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

        # ekfzladeprofile.data_path() automatically prefixes this relative
        # directory with <emob_cold_profile>/data.
        self.profile_folder_name = str(
            Path(f"data_{self.cfg.date_input}")
            / self.cfg.net_name
            / self.cfg.station_name
            / "ev"
        )

        # Absolute path, mainly for logging and directory creation.
        self.ev_dir = Path(ekfz_data_path(self.profile_folder_name))
        self.ev_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        mob_groups_short: Sequence[int],
        ch_refuel_list: Sequence[float],
        veh_per_region: int = 10000,
        njobs: int = 7,
    ) -> None:
        """Generate mixed charging/SOC profiles for multiple thresholds.

        Parameters
        ----------
        mob_groups_short:
            Mobility groups for which ``ch_avail`` profiles were generated.
        ch_refuel_list:
            Refuelling thresholds as fractions of battery capacity, for
            example ``[0.4, 0.6, 0.9]``.
        veh_per_region:
            Number of profiles sampled per region group.
        njobs:
            Number of worker threads used when Numba is unavailable. The
            threading backend is intentional because all workers must see the
            temporarily changed ``CH_REFUEL_THRESHOLD`` value.
        """
        mob_groups = [int(group) for group in mob_groups_short]
        if not mob_groups:
            raise ValueError("mob_groups_short must contain at least one mobility group.")

        if not ch_refuel_list:
            raise ValueError("ch_refuel_list must contain at least one threshold.")

        worker_count = max(1, int(njobs))
        scenario = {
            # Kalenderjahr and Wetterjahr remain defined in
            # ekfzladeprofile/Input_Ladeprofile.xlsx.
            "Szenariojahr": int(self.cfg.scenario_year),
        }

        print(f"Reading and writing ekfzladeprofile data in: {self.ev_dir}")

        # A process-based backend would import a fresh SOC module in every
        # worker and therefore lose the temporary threshold. Threads share the
        # same module state. If Numba is installed, ekfzladeprofile itself
        # switches the SOC calculation to one process.
        with parallel_backend("threading", n_jobs=worker_count):
            for threshold in ch_refuel_list:
                threshold_value = float(threshold)
                if not 0.0 <= threshold_value <= 1.0:
                    raise ValueError(
                        "Each ch_refuel_threshold must lie between 0 and 1; "
                        f"received {threshold!r}."
                    )

                # Keep the spelling supplied by ch_refuel_list so later stages
                # can construct the same filename, e.g. 0.4 or 1.
                threshold_label = str(threshold)
                profile_output = f"ch_profiles_{threshold_label}"

                print(
                    "Generating mixed profiles with "
                    f"ch_refuel_threshold={threshold_value}"
                )

                with temporary_ch_refuel_threshold(threshold_value):
                    mix_profiles(
                        data_folder_name=self.profile_folder_name,
                        mob_groups=mob_groups,
                        date_input=self.cfg.date_input,
                        profile_input="ch_avail",
                        profile_output=profile_output,
                        country="DEU",
                        random_seed=self.cfg.seed,
                        veh_per_region=int(veh_per_region),
                        parallel=worker_count > 1,
                        njobs=worker_count,
                        split_size=500,
                        scenario=scenario,
                    )