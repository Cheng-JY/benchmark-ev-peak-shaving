import matplotlib.pyplot as plt
import numpy as np

def plot_avail_charging_scatter(
    results_dict:dict[float, list[float]],
    n_ev: int,
    filename: str = "ch_avail_per_ev_scatter",
): 
    plt.figure()

    for thr, durations in results_dict.items():
        xs = np.full_like(durations, thr)
        jitter = np.random.uniform(-0.005, 0.005, size=len(durations))
        plt.scatter(xs + jitter, durations, s=10, alpha=0.6, edgecolor='k', linewidth=0.3)

    plt.xticks(list(results_dict.keys()))
    plt.xlabel("ch_refuel_threshold")
    plt.ylabel("Plug in duration (h/month)")
    plt.title(f"Plug in duration vs ch_refuel_threshold home charing ({n_ev} evs)")
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')

def plot_avail_charging_boxplot(
    results_dict:dict[float, list[float]],
    n_ev: int,
    filename: str = "ch_avail_per_ev_boxplot",
): 
    plt.figure()

    plt.boxplot(results_dict.values())
    plt.xticks([i for i in range(1, 12)], list(results_dict.keys()))
    plt.xlabel("ch_refuel_threshold")
    plt.ylabel("Plug in duration (h/month)")
    plt.title(f"Plug in duration vs ch_refuel_threshold home charing ({n_ev} evs)")
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')


def plot_annual_mileage_ch_avail_scatter(
    annual_mileage: list[float],
    ch_avail_hours: list[float],
    ch_refuel_threshold: float,
    filename: str = "annual_mileage_ch_avail_scatter",
): 
    plt.figure()
    plt.scatter(annual_mileage, ch_avail_hours, s=10, facecolor='none', edgecolors='blue', label="Data")
    plt.xlabel("Annual mileage (km)")
    plt.ylabel("Plug in duration (h/month)")
    plt.title(f"Annual mileage vs Plug in duration home charing ({ch_refuel_threshold})")
    plt.savefig(f"{filename}_{ch_refuel_threshold}.pdf", bbox_inches='tight')

