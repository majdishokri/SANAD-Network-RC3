# GPU device-specific port of sionna_gen.py, source hashed in v34 frozen manifests.
"""
Generate realistic per-tower covariance reports with NVIDIA Sionna's implementation of the
3GPP TR 38.901 Urban Macro (UMa) channel model (the industry-standard 5G channel model).

7 base stations on a hexagonal grid (inter-site distance 500 m, height 25 m), each with an
8-element horizontal array (3GPP 38.901 element pattern, 3.5 GHz, half-wavelength spacing),
all sectors facing east. Each reporting period, a fresh set of 30 users is dropped in the
area (outdoor/indoor mix, pedestrian and vehicle speeds). Every tower receives all users'
uplink signals through its OWN 38.901 channel: own path loss, shadowing, LoS/NLoS state,
clusters and angles. So, unlike the textbook model, every tower sees a DIFFERENT field.

Per tower and period we form the received spatial covariance over 12 subcarriers of a
20 MHz carrier, normalise by the receiver noise power, and draw a finite-snapshot estimate
(400 snapshots); each report averages 4 independent drops of 30 users, exactly what a tower's own estimator would report.

Output: npz with R[seed][t, tower, M, M] (healthy, fault-free). Faults, glitches and
missing reports are applied later by the evaluation script.
"""
import sys
import time
import numpy as np
import torch
from sionna.phy.channel.tr38901 import PanelArray, UMa
from sionna.phy.channel import cir_to_ofdm_channel

DEVICE = "cuda:0"
from geometry_config import N_ARRAYS, N_ELEMENTS
M, N_BS, N_UT, N_SNAP = N_ELEMENTS, N_ARRAYS, 30, 400
K_DROPS = 4                            # independent user drops averaged into one 15-min report
FC = 3.5e9
ISD = 500.0
UT_POWER_W = 0.2                       # 23 dBm
NOISE_W = 1.380649e-23 * 290 * 20e6 * 10 ** (5 / 10)   # kTB, 20 MHz, 5 dB noise figure

bs_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=M, polarization="single",
                      polarization_type="V", antenna_pattern="38.901", carrier_frequency=FC, device=DEVICE)
ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization="single",
                      polarization_type="V", antenna_pattern="omni", carrier_frequency=FC, device=DEVICE)
model = UMa(carrier_frequency=FC, o2i_model="low", ut_array=ut_array, bs_array=bs_array,
            direction="uplink", enable_pathloss=True, enable_shadow_fading=True, device=DEVICE)

# Keep the original 7 sites exactly; extend the same hexagonal lattice as needed.
def _sites(n):
    points=[np.array([0.0,0.0])]
    ring=1
    while len(points)<n:
        corners=[ring*ISD*np.array([np.cos(np.pi/3*k+np.pi/6),
                                     np.sin(np.pi/3*k+np.pi/6)]) for k in range(6)]
        for side in range(6):
            for step in range(ring):
                if len(points)>=n:break
                f=step/ring
                points.append((1-f)*corners[side]+f*corners[(side+1)%6])
        ring+=1
    return np.asarray(points)
BS_XY=_sites(N_BS)
FREQS = torch.tensor(np.linspace(-9e6, 9e6, 12), dtype=torch.float32, device=DEVICE)


def drop(rng, batch):
    """Users uniformly in a disc covering the 7 cells, >= 35 m from every site."""
    pts = []
    while len(pts) < batch * N_UT:
        r = (ISD if N_BS <= 7 else max(ISD,np.linalg.norm(BS_XY,axis=1).max())) * 1.3 * np.sqrt(rng.random()); th = 2 * np.pi * rng.random()
        p = np.array([r * np.cos(th), r * np.sin(th)])
        if np.min(np.linalg.norm(BS_XY - p, axis=1)) >= 35:
            pts.append(p)
    xy = np.array(pts).reshape(batch, N_UT, 2)
    ut_loc = np.concatenate([xy, np.full((batch, N_UT, 1), 1.5)], -1)
    bs_loc = np.tile(np.concatenate([BS_XY, np.full((N_BS, 1), 25.0)], -1), (batch, 1, 1))
    speed = rng.choice([1.0, 3.0, 15.0], size=(batch, N_UT))
    ang = rng.uniform(0, 2 * np.pi, (batch, N_UT))
    vel = np.stack([speed * np.cos(ang), speed * np.sin(ang), np.zeros_like(speed)], -1)
    ut_ori = np.stack([rng.uniform(0, 2 * np.pi, (batch, N_UT)), np.zeros((batch, N_UT)), np.zeros((batch, N_UT))], -1)
    bs_ori = np.zeros((batch, N_BS, 3))
    bs_ori[..., 1] = np.radians(10)     # 10 degree downtilt
    indoor = rng.random((batch, N_UT)) < 0.5
    T = lambda x: torch.tensor(x, dtype=torch.float32, device=DEVICE)
    return T(ut_loc), T(bs_loc), T(ut_ori), T(bs_ori), T(vel), torch.tensor(indoor, device=DEVICE)


def expected_cov(rng, batch):
    ut_loc, bs_loc, ut_ori, bs_ori, vel, indoor = drop(rng, batch)
    model.set_topology(ut_loc, bs_loc, ut_ori, bs_ori, vel, indoor)
    a, tau = model(num_time_samples=1, sampling_frequency=1.0)
    h = cir_to_ofdm_channel(FREQS, a, tau, normalize=False)
    # h: [batch, num_rx=BS, rx_ant, num_tx=UT, tx_ant=1, time=1, F]
    h = h[:, :, :, :, 0, 0, :].detach().cpu().numpy()                       # [b, BS, M, UT, F]
    h = np.transpose(h, (0, 1, 4, 3, 2))                       # [b, BS, F, UT, M]
    hw = h * np.sqrt(UT_POWER_W / NOISE_W)                     # noise-normalised
    return np.einsum("bsfum,bsfun->bsmn", hw, hw.conj()) / len(FREQS)


def covariances(rng, batch):
    # a 15-minute report sees many users: average K independent drops, add receiver noise
    R = sum(expected_cov(rng, batch) for _ in range(K_DROPS)) / K_DROPS + np.eye(M)[None, None]
    # finite-snapshot estimate per tower: x = R^(1/2) z
    out = np.empty_like(R)
    for b in range(batch):
        for s in range(N_BS):
            w, V = np.linalg.eigh(R[b, s])
            Rh = (V * np.sqrt(np.maximum(w, 0))) @ V.conj().T
            Z = (rng.standard_normal((M, N_SNAP)) + 1j * rng.standard_normal((M, N_SNAP))) / np.sqrt(2)
            X = Rh @ Z
            out[b, s] = X @ X.conj().T / N_SNAP
    return out


def main():
    periods = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [100, 101, 200, 201, 202, 203, 204, 205]
    out_name = sys.argv[3] if len(sys.argv) > 3 else "sionna_reports.npz"
    data = {}
    for sd in seeds:
        torch.manual_seed(sd)
        rng = np.random.default_rng(sd)
        t0 = time.time()
        chunks = [covariances(rng, 50) for _ in range(periods // 50)]
        data[f"s{sd}"] = np.concatenate(chunks)
        print(f"seed {sd}: {data[f's{sd}'].shape} in {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out_name, **data)


if __name__ == "__main__":
    main()
