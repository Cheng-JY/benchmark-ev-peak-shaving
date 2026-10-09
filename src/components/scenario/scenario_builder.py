"""Scenario builder utilities for the EV/V2G analysis workflow."""

# jcheng/compoenents/scenario/scenario_builder.py

from ecv2g.db_api.mongodb import get_task_document
from ecv2g.scenario.scenario_workflow import run_scenario_workflow
import pandas as pd
from ...utils._data import data_path
from .config import ScenarioConfig
from pathlib import Path

class ScenarioData:

    """Represent scenario data and its associated operations."""
    SUBDIRS = [
        "household",
        "heat_pump",
        "ev",
        "aggregation",
        "optimization",
        "regression",
        "analysis",
    ]
    
    def __init__(self, date_input, net_name, station_name, output_folder=None):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        date_input : object
            Date input used by this operation.
        net_name : object
            Net name used by this operation.
        station_name : object
            Station name used by this operation.
        output_folder : object
            Output folder used by this operation."""
        if output_folder is None:
            base = Path(data_path(f"data_{date_input}"))
            self.folder = base / net_name / station_name
        else:
            self.folder = Path(output_folder).expanduser().resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
    
        for name in self.SUBDIRS:
            d = self.folder / name
            d.mkdir(parents=True, exist_ok=True)
            setattr(self, f"{name}_dir", d)

    def __repr__(self):
        """Return the textual representation of the instance.
        
        Returns
        -------
        object
            Result computed as ``f'<ScenarioData folder={self.folder}>'``."""
        return f"<ScenarioData folder={self.folder}>"



class ScenarioBuilder:
    """Represent scenario builder and its associated operations."""
    def __init__(
        self,
        scenario_config: ScenarioConfig,
        output_folder=None,
        include_grid_precheck: bool = False,
        on_conflict: str = "keep",
    ):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        scenario_config : ScenarioConfig
            Scenario config used by this operation.
        output_folder : object
            Output folder used by this operation.
        include_grid_precheck : bool
            Include grid precheck used by this operation.
        on_conflict : str
            On conflict used by this operation."""
        self.cfg = scenario_config
        self.output_folder = output_folder
        self.include_grid_precheck = include_grid_precheck
        self.on_conflict = on_conflict

    def _load_and_export_document(self, scenario_hash):
        """Load and export document.
        
        Parameters
        ----------
        scenario_hash : object
            Scenario hash used by this operation.
        
        Returns
        -------
        object
            Result computed as ``net_pop_data``.
        
        Raises
        ------
        RuntimeError
            If an explicit validation or processing check in this operation fails.
        KeyError
            If an explicit validation or processing check in this operation fails."""
        document = get_task_document({"scenario_hash": scenario_hash})
        if document is None:
            raise RuntimeError(
                f"No scenario_tasks document found for hash {scenario_hash!r}."
            )
        required = ("net_pop_data", "net_building_data", "hh_profile")
        missing = [key for key in required if key not in document]
        if missing:
            raise KeyError(
                f"Scenario document {scenario_hash!r} is missing: {missing}"
            )

        net_pop_data = pd.DataFrame(document["net_pop_data"])
        net_building_data = pd.DataFrame(document["net_building_data"])
        net_pop_data.to_csv(
            self.scenario_data.folder / "net_pop_data.csv",
            index=False,
            sep=";",
        )
        net_building_data.to_csv(
            self.scenario_data.folder / "net_building_data.csv",
            index=False,
            sep=";",
        )
        self.net_building_data = net_building_data
        self.scenario_document = document
        return net_pop_data

    def build(self):
        """Build .
        
        Returns
        -------
        object
            Result computed as ``net_pop_data``."""
        cfg = self.cfg
        scenario_data = ScenarioData(
            cfg.date_input,
            cfg.net_name,
            cfg.station_name,
            output_folder=self.output_folder,
        )
        folder = scenario_data.folder
        self.scenario_data = scenario_data
        
        # The current workflow is building-based, writes the complete scenario
        # document to scenario_tasks and returns only its hash.
        result = run_scenario_workflow(
            net_name=cfg.net_name,
            station_name=cfg.station_name,
            seed=cfg.seed,
            scenario_hp_ratio=cfg.scenario_hp_ratio,
            scenario_ev_ratio=cfg.scenario_ev_ratio,
            cp_double_ev_ratio=cfg.cp_double_ev_ratio,
            income_weight=cfg.income_weight,
            vehicle_count_weight=cfg.vehicle_count_weight,
            efh_weight=cfg.efh_weight,
            garage_weight=cfg.garage_weight,
            include_grid_precheck=self.include_grid_precheck,
            on_conflict=self.on_conflict,
        )
        scenario_hash = result[0] if isinstance(result, tuple) else result
        self.cfg.scenario_hash = scenario_hash
        net_pop_data = self._load_and_export_document(scenario_hash)
        cfg.save_yaml(folder / "config.yaml")
        return net_pop_data
    
    def get_scenario(self):
        """Return scenario.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        cfg = self.cfg
        
        if cfg.scenario_hash is None:
            return self.build()
        else:
            scenario_data = ScenarioData(
                cfg.date_input,
                cfg.net_name,
                cfg.station_name,
                output_folder=self.output_folder,
            )
            self.scenario_data = scenario_data
            return self._load_and_export_document(cfg.scenario_hash)

        """ from src.components.scenario.config import ScenarioConfig
            from src.components.scenario.scenario_builder import ScenarioBuilder

            cfg = ScenarioConfig(
                date_input="20250111",
                net_name="uw_langenselbold",
                station_name="ST. Waldstr. 630KVATR",
                scenario_hp_ratio=0.4,         # ←你可以随时调
                scenario_ev_ratio=0.35,        # ←随时调
                cp_double_ev_ratio=0.25,       # ←随时调
            )

            sb = ScenarioBuilder(cfg)
            net_pop_data = sb.build()
        """
