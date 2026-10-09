"""Config utilities for the EV/V2G analysis workflow."""

from dataclasses import dataclass, field, asdict
import yaml
import json
from pathlib import Path

@dataclass
class ScenarioConfig:
    # --- Basic Scenario Information ---
    """Represent scenario config and its associated operations."""
    date_input: str
    net_name: str
    station_name: str
    region_name: str
    seed: int = 13
    scenario_year: int = 2030

    # --- Heat Pump Assignment Parameters ---
    existing_hp_ratio: float = 0.0
    scenario_hp_ratio: float = 0.3

    # --- EV Charging Point Assignment Parameters ---
    scenario_ev_ratio: float = 0.8
    cp_double_ev_ratio: float = 0.2
    income_weight: float = 1.0
    vehicle_count_weight: float = 1.0
    efh_weight: float = 1.5
    garage_weight: float = 1.2

    # --- export ---
    export_paths: bool = True

    # ----ScenarioHash----
    scenario_hash: str|None = None

    # --- Methods to load/save config ---
    @staticmethod
    def from_dict(d: dict):
        """From dict.
        
        Parameters
        ----------
        d : dict
            D used by this operation.
        
        Returns
        -------
        object
            Result computed as ``ScenarioConfig(**d)``."""
        return ScenarioConfig(**d)
    
    @staticmethod
    def from_yaml(file_path: str):
        """From yaml.
        
        Parameters
        ----------
        file_path : str
            File path used by this operation.
        
        Returns
        -------
        object
            Result computed as ``ScenarioConfig(**data)``."""
        with open(file_path, 'r', encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return ScenarioConfig(**data)
    
    def from_json(file_path: str):
        """From json.
        
        Parameters
        ----------
        file_path : str
            File path used by this operation.
        
        Returns
        -------
        object
            Result computed as ``ScenarioConfig(**data)``."""
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return ScenarioConfig(**data)
    
    def save_yaml(self, file_path: Path):
        """
        Saves the current configuration instance to a YAML file.
        
        Args:
            file_path (str): The path where the YAML file will be saved.
        """
        # Convert the dataclass to a dictionary
        data_dict = asdict(self)
        
        # Ensure the parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to YAML file
        with open(file_path, 'w', encoding="utf-8") as f:
            yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"Configuration saved to: {file_path}") 
