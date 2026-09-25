"""Use repository maps directly, without modifying the installed PySC2 package."""
from pathlib import Path

from pysc2.maps.lib import Map
from src.sc2env.config import get_map_config


class BundledSAGEMap(Map):
    players = 2
    step_mul = 10

    def __init__(self, scenario):
        config = get_map_config(scenario)[1]
        self.directory = str(Path(__file__).resolve().parents[2] / "assets/maps")
        self.filename = config["map_name"]
        self._name = self.filename
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"Missing SAGE map: {self.path}. Retrieve the Git LFS assets.")

    @property
    def name(self):
        return self._name

    def data(self, run_config):
        return Path(self.path).read_bytes()
