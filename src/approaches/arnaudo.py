import pandas as pd
import numpy as np

class Arnaudo():
    def __init__(self, start_charging, stop_charging, start_discharging):
        self.start_charging = start_charging
        self.stop_charging = stop_charging
        self.start_discharging = start_discharging

    def charging(self):
        """
        if SoC < 50%, start charging, until 90% stop
        if SoC > 80%, start discharging, until 80% stop
        :return:
        """
        pass

