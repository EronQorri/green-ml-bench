import threading
import time

CARBON_KG_PER_KWH = 0.485  # German grid ~485 gCO2/kWh

_computer = None

def _init_computer():
    global _computer
    if _computer is not None:
        return
    try:
        from HardwareMonitor.Hardware import Computer, IVisitor

        class UpdateVisitor(IVisitor):
            __namespace__ = "ThesisMonitor"
            def VisitComputer(self, c): c.Traverse(self)
            def VisitHardware(self, h):
                h.Update()
                for sub in h.SubHardware:
                    sub.Update()
            def VisitParameter(self, p): pass
            def VisitSensor(self, s): pass

        _computer = Computer()
        _computer.IsCpuEnabled = True
        _computer.Open()
        _computer._visitor = UpdateVisitor()
    except Exception as e:
        print(f"[PowerMonitor] Init fehlgeschlagen: {e}")


def _get_cpu_package_power():
    _init_computer()
    if _computer is None:
        return None
    try:
        from HardwareMonitor.Hardware import SensorType
        _computer.Accept(_computer._visitor)
        for hw in _computer.Hardware:
            for sensor in hw.Sensors:
                if "Package" in str(sensor.Name) and sensor.SensorType == SensorType.Power:
                    val = sensor.Value
                    return float(val) if val is not None else None
    except Exception as e:
        print(f"[PowerMonitor] Lesefehler: {e}")
    return None


class CPUPowerMonitor:
    """
    Misst CPU Package Power parallel zu einem Experiment.
    Nutzung:
        monitor = CPUPowerMonitor(interval=0.5)
        monitor.start()
        # ... dein Training ...
        result = monitor.stop()  # {"energy_wh": ..., "avg_watt": ..., "duration_s": ...}
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._readings = []
        self._running = False
        self._thread = None
        self._start_time = None

    def _poll(self):
        while self._running:
            w = _get_cpu_package_power()
            if w is not None:
                self._readings.append(w)
            time.sleep(self.interval)

    def start(self):
        self._readings = []
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._running = False
        self._thread.join()
        duration_s = time.time() - self._start_time

        if not self._readings:
            return {"energy_wh": None, "avg_watt": None, "duration_s": duration_s}

        avg_watt = sum(self._readings) / len(self._readings)
        energy_wh = avg_watt * (duration_s / 3600)

        return {
            "energy_wh": energy_wh,
            "energy_kg_co2": energy_wh * CARBON_KG_PER_KWH / 1000,
            "avg_watt": avg_watt,
            "duration_s": duration_s,
            "n_samples": len(self._readings),
        }


def compute_corrected_co2(tracker, cpu_result):
    """CO2 mit HardwareMonitor-CPU statt CodeCarbon-CPU-Schätzung."""
    data = tracker.final_emissions_data
    gpu_ram_kwh = (data.gpu_energy or 0.0) + (data.ram_energy or 0.0)
    cpu_kwh = cpu_result["energy_wh"] / 1000.0
    return (gpu_ram_kwh + cpu_kwh) * CARBON_KG_PER_KWH


def print_cpu_summary(cpu_result, cc_cpu_energy_kwh=None):
    print(f"\n--- CPU Power Monitor ---")
    print(f"Avg CPU power:  {cpu_result['avg_watt']:.2f} W")
    print(f"CPU energy:     {cpu_result['energy_wh']:.4f} Wh")
    print(f"Duration:       {cpu_result['duration_s']:.2f} s")
    print(f"Samples taken:  {cpu_result['n_samples']}")
    if cc_cpu_energy_kwh is not None:
        print(f"CodeCarbon CPU estimate: {cc_cpu_energy_kwh * 1000:.4f} Wh")
    print(f"-------------------------\n")