import math, random
import os, time
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------------
# Config
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_COLLOC_PDE  = 4096
N_COLLOC_NORM = 4096   # separate batch for normalization (uniform)
LR = 2e-3
EPOCHS_ADAM = 50000
USE_LBFGS = True
SEED = 1

WIDTH = 128
DEPTH = 6
FOURIER_K = 1
DOMAIN = (0.0, 2*math.pi)

# -------------------------------------------------
# Oversampling controls (physical domain sampling)
# -------------------------------------------------
USE_STAGNATION_OVERSAMPLE = False  # kept off for the sweep: we care about where λ converges, not speed
FOCUS_FRAC = 0.15      # 15% focused near stagnation points
FOCUS_SIGMA = 0.25     # radians; try 0.15–0.35

# -------------------------------------------------
# (these are NOT used in the sweep; per-run init overwrites them)
# -------------------------------------------------
INIT_LAMR = 1
INIT_LAMI = 1

# -------------------------------------------------
# Loss weights
# -------------------------------------------------
W_PDE   = 10.0
W_NORM  = 5.0     # soft omega-normalization weight
W_PHASE = 1e-3
W_MEAN  = 1e-2
W_RQ    = 1.0     # multiplies L_rq (which has its own 1e-3 scale)

# -------------------------------------------------
# Saving
# -------------------------------------------------
SAVE_DIR = "out_lambda_sweep"
RUN_TAG = time.strftime("%Y%m%d_%H%M%S")

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_all_seeds(SEED)

def sample_xy_uniform(n):
    x1 = torch.rand(n, 1, device=DEVICE) * (DOMAIN[1]-DOMAIN[0]) + DOMAIN[0]
    x2 = torch.rand(n, 1, device=DEVICE) * (DOMAIN[1]-DOMAIN[0]) + DOMAIN[0]
    return x1.requires_grad_(True), x2.requires_grad_(True)

def sample_xy_mixture(n, frac_focus=FOCUS_FRAC, sigma=FOCUS_SIGMA):
    """
    Mixture sampler:
      - (1-frac_focus) uniform in [0,2π]^2
      - frac_focus concentrated near hyperbolic stagnation points (wrapped Gaussian)

    Hyperbolic stagnation points of u_s = (-cos x sin y, sin x cos y):
      (π/2, π/2), (π/2, 3π/2), (3π/2, π/2), (3π/2, 3π/2)
    """
    if (not USE_STAGNATION_OVERSAMPLE) or frac_focus <= 0.0:
        return sample_xy_uniform(n)

    n_focus = int(frac_focus * n)
    n_uni = n - n_focus

    # Uniform part
    x1u = torch.rand(n_uni, 1, device=DEVICE) * (DOMAIN[1]-DOMAIN[0]) + DOMAIN[0]
    x2u = torch.rand(n_uni, 1, device=DEVICE) * (DOMAIN[1]-DOMAIN[0]) + DOMAIN[0]

    # Focused part
    if n_focus > 0:
        pts = torch.tensor([
            [math.pi/2, math.pi/2],
            [math.pi/2, 3*math.pi/2],
            [3*math.pi/2, math.pi/2],
            [3*math.pi/2, 3*math.pi/2],
        ], device=DEVICE, dtype=torch.float32)

        idx = torch.randint(0, pts.shape[0], (n_focus,), device=DEVICE)
        centers = pts[idx]
        noise = sigma * torch.randn(n_focus, 2, device=DEVICE)
        xf = centers + noise

        # wrap to [0,2π)
        x1f = torch.remainder(xf[:, 0:1], 2*math.pi)
        x2f = torch.remainder(xf[:, 1:2], 2*math.pi)

        x1 = torch.cat([x1u, x1f], dim=0)
        x2 = torch.cat([x2u, x2f], dim=0)
    else:
        x1, x2 = x1u, x2u

    # Shuffle
    perm = torch.randperm(n, device=DEVICE)
    x1 = x1[perm].requires_grad_(True)
    x2 = x2[perm].requires_grad_(True)
    return x1, x2

def per_features(x1, x2, K=FOURIER_K):
    feats = []
    for k in range(1, K+1):
        feats += [torch.sin(k*x1), torch.cos(k*x1),
                  torch.sin(k*x2), torch.cos(k*x2)]
    return torch.cat(feats, dim=1)

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True
    )[0]

def laplacian(scalar, x1, x2):
    ds_dx1 = grad(scalar, x1)
    ds_dx2 = grad(scalar, x2)
    d2s_dx1 = grad(ds_dx1, x1)
    d2s_dx2 = grad(ds_dx2, x2)
    return d2s_dx1 + d2s_dx2

# -------------------------------------------------
# Base fields (2D Taylor–Green stationary solution)
# -------------------------------------------------
def us_omega_gradomega(x1, x2):
    us1 = -torch.cos(x1)*torch.sin(x2)
    us2 =  torch.sin(x1)*torch.cos(x2)
    omega_s = 2.0*torch.cos(x1)*torch.cos(x2)
    domg_dx1 = -2.0*torch.sin(x1)*torch.cos(x2)
    domg_dx2 = -2.0*torch.cos(x1)*torch.sin(x2)
    return (us1, us2), omega_s, (domg_dx1, domg_dx2)

# -------------------------------------------------
# Network
# -------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=128, depth=5):
        super().__init__()
        layers = []
        dims = [in_dim] + [width]*depth + [out_dim]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.SiLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class EigenPINN(nn.Module):
    def __init__(self, K=1, width=128, depth=5):
        super().__init__()
        self.model = MLP(4*K, 2, width, depth)   # -> (psi_R, psi_I)

        # lambda
        self.lamR = nn.Parameter(torch.tensor(float(INIT_LAMR), dtype=torch.float32))
        self.lamI = nn.Parameter(torch.tensor(float(INIT_LAMI), dtype=torch.float32))

    @property
    def lambda_R(self):
        return self.lamR

    @property
    def lambda_I(self):
        return self.lamI

    def forward(self, x1, x2):
        feats = per_features(x1, x2, K=FOURIER_K)
        psi_R, psi_I = self.model(feats).chunk(2, dim=1)
        return psi_R, psi_I

# -------------------------------------------------
# Loss
# -------------------------------------------------
def loss_on_points(model, x1_pde, x2_pde, x1_norm, x2_norm):
    # PDE batch
    psi_R, psi_I = model(x1_pde, x2_pde)

    lap_psi_R = laplacian(psi_R, x1_pde, x2_pde)
    lap_psi_I = laplacian(psi_I, x1_pde, x2_pde)
    omg_R = -lap_psi_R
    omg_I = -lap_psi_I

    domgR_dx1 = grad(omg_R, x1_pde); domgR_dx2 = grad(omg_R, x2_pde)
    domgI_dx1 = grad(omg_I, x1_pde); domgI_dx2 = grad(omg_I, x2_pde)

    dpsiR_dx1 = grad(psi_R, x1_pde); dpsiR_dx2 = grad(psi_R, x2_pde)
    dpsiI_dx1 = grad(psi_I, x1_pde); dpsiI_dx2 = grad(psi_I, x2_pde)
    uR1, uR2 = dpsiR_dx2, -dpsiR_dx1
    uI1, uI2 = dpsiI_dx2, -dpsiI_dx1

    (us1, us2), omega_s, (domegas_dx1, domegas_dx2) = us_omega_gradomega(x1_pde, x2_pde)

    us_dot_grad_omg_R = us1*domgR_dx1 + us2*domgR_dx2
    us_dot_grad_omg_I = us1*domgI_dx1 + us2*domgI_dx2
    upr_dot_grad_omegas_R = uR1*domegas_dx1 + uR2*domegas_dx2
    upr_dot_grad_omegas_I = uI1*domegas_dx1 + uI2*domegas_dx2

    lamR = model.lambda_R
    lamI = model.lambda_I

    # Euler (nu=0) eigen-vorticity residual
    Rw_R = (lamR*omg_R - lamI*omg_I) + us_dot_grad_omg_R + upr_dot_grad_omegas_R
    Rw_I = (lamR*omg_I + lamI*omg_R) + us_dot_grad_omg_I + upr_dot_grad_omegas_I
    L_pde = (Rw_R.pow(2) + Rw_I.pow(2)).mean()

    # Soft omega-normalization on separate UNIFORM batch (keeps normalization unbiased)
    psi_Rn, psi_In = model(x1_norm, x2_norm)
    omg_Rn = -laplacian(psi_Rn, x1_norm, x2_norm)
    omg_In = -laplacian(psi_In, x1_norm, x2_norm)
    mean_omg2 = (omg_Rn.pow(2) + omg_In.pow(2)).mean()
    L_norm_omg = (mean_omg2 - 1.0)**2

    # Phase anchor: psi_I(0,0)=0
    x0 = torch.zeros((1,1), device=DEVICE, requires_grad=True)
    y0 = torch.zeros((1,1), device=DEVICE, requires_grad=True)
    psiR0, psiI0 = model(x0, y0)
    L_phase = psiI0.pow(2).mean()

    # Mean-zero gauge (use uniform batch)
    L_mean = psi_Rn.mean().pow(2) + psi_In.mean().pow(2)

    # Rayleigh-quotient consistency (on PDE batch)
    rhsR = -(us_dot_grad_omg_R + upr_dot_grad_omegas_R)
    rhsI = -(us_dot_grad_omg_I + upr_dot_grad_omegas_I)
    numR = (omg_R*rhsR + omg_I*rhsI).mean()
    numI = (omg_R*rhsI - omg_I*rhsR).mean()
    den  = (omg_R.pow(2)+omg_I.pow(2)).mean() + 1e-12
    rqR, rqI = numR/den, numI/den
    L_rq = 1e-3*((lamR - rqR).pow(2) + (lamI - rqI).pow(2))

    loss = W_PDE*L_pde + W_NORM*L_norm_omg + W_PHASE*L_phase + W_MEAN*L_mean + W_RQ*L_rq

    parts = dict(
        L_pde=float(L_pde.detach()),
        mean_omg2=float(mean_omg2.detach()),
        L_norm=float(L_norm_omg.detach()),
        L_phase=float(L_phase.detach()),
        L_mean=float(L_mean.detach()),
        L_rq=float(L_rq.detach()),
        lamR=float(lamR.detach()),
        lamI=float(lamI.detach()),
        rqR=float(rqR.detach()),
        rqI=float(rqI.detach()),
    )
    return loss, parts

def loss_fn(model):
    # PDE points: oversampled mixture
    x1_pde,  x2_pde  = sample_xy_mixture(N_COLLOC_PDE, frac_focus=FOCUS_FRAC, sigma=FOCUS_SIGMA)
    # Norm points: uniform (important to avoid bias from oversampling)
    x1_norm, x2_norm = sample_xy_uniform(N_COLLOC_NORM)
    return loss_on_points(model, x1_pde, x2_pde, x1_norm, x2_norm)

# -------------------------------------------------
#
# Sweep helpers (9-point grid in [-1,1]^2 with epsilon-epsilon center)
# -------------------------------------------------
def train_one_run(init_lamR, init_lamI, run_dir, seed_offset=0):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Keep sampling identical across runs
    set_all_seeds(SEED + seed_offset)

    model = EigenPINN(K=FOURIER_K, width=WIDTH, depth=DEPTH).to(DEVICE)
    with torch.no_grad():
        model.lamR.copy_(torch.tensor(float(init_lamR), device=model.lamR.device))
        model.lamI.copy_(torch.tensor(float(init_lamI), device=model.lamI.device))

    opt = optim.Adam(model.parameters(), lr=LR)

    for it in range(EPOCHS_ADAM):
        opt.zero_grad(set_to_none=True)
        loss, parts = loss_fn(model)
        loss.backward()
        opt.step()

        if (it+1) % 500 == 0:
            lamR = model.lambda_R.item()
            lamI = model.lambda_I.item()
            abs_lam = (lamR**2 + lamI**2) ** 0.5
            print(
                f"[init=({init_lamR:+.3f},{init_lamI:+.3f}i) Adam {it+1}] "
                f"loss={loss.item():.3e} lam=({lamR:.6f},{lamI:+.6f}i) |lam|={abs_lam:.3e} "
                f"RQ=({parts['rqR']:.6f},{parts['rqI']:+.6f}i) "
                f"L_norm={parts['L_norm']:.2e} mean|omg|^2={parts['mean_omg2']:.3e} "
                f"PDE={parts['L_pde']:.2e} L_rq={parts['L_rq']:.2e}"
            )

    if USE_LBFGS:
        x1_pde_fix, x2_pde_fix   = sample_xy_mixture(N_COLLOC_PDE, frac_focus=FOCUS_FRAC, sigma=FOCUS_SIGMA)
        x1_norm_fix, x2_norm_fix = sample_xy_uniform(N_COLLOC_NORM)
        x1_pde_fix  = x1_pde_fix.detach().requires_grad_(True)
        x2_pde_fix  = x2_pde_fix.detach().requires_grad_(True)
        x1_norm_fix = x1_norm_fix.detach().requires_grad_(True)
        x2_norm_fix = x2_norm_fix.detach().requires_grad_(True)

        opt_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=6000,
            tolerance_grad=1e-7,
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe"
        )

        def closure():
            opt_lbfgs.zero_grad(set_to_none=True)
            loss, _ = loss_on_points(model, x1_pde_fix, x2_pde_fix, x1_norm_fix, x2_norm_fix)
            loss.backward()
            return loss

        opt_lbfgs.step(closure)
        loss, parts = loss_on_points(model, x1_pde_fix, x2_pde_fix, x1_norm_fix, x2_norm_fix)

        print(
            f"[LBFGS init=({init_lamR:+.3f},{init_lamI:+.3f}i)] final loss={loss.item():.3e} "
            f"lam=({parts['lamR']:.6f},{parts['lamI']:+.6f}i) "
            f"RQ=({parts['rqR']:.6f},{parts['rqI']:+.6f}i) "
            f"L_norm={parts['L_norm']:.2e} mean|omg|^2={parts['mean_omg2']:.3e} PDE={parts['L_pde']:.2e}"
        )
    else:
        loss, parts = loss_fn(model)

    lamR = model.lambda_R.item()
    lamI = model.lambda_I.item()
    abs_lam = (lamR**2 + lamI**2) ** 0.5

    return {
        "init_lamR": float(init_lamR),
        "init_lamI": float(init_lamI),
        "final_lamR": float(lamR),
        "final_lamI": float(lamI),
        "final_abs_lam": float(abs_lam),
        "final_loss": float(loss.item()),
        "final_L_pde": float(parts["L_pde"]),
        "final_mean_omg2": float(parts["mean_omg2"]),
        "final_L_norm": float(parts["L_norm"]),
        "final_L_rq": float(parts["L_rq"]),
        "final_rqR": float(parts["rqR"]),
        "final_rqI": float(parts["rqI"]),
        "run_dir": str(run_dir),
    }

# -------------------------------------------------
# Main: 9-point grid sweep for lambda init
# -------------------------------------------------
def main():
    base = Path(SAVE_DIR) / f"sweep_{RUN_TAG}"
    base.mkdir(parents=True, exist_ok=True)

    # 9-point grid in [-1,1]x[-1,1] with epsilon-epsilon "center"
    eps = 0.02
    re_vals = [-1.0, eps, 1.0]
    im_vals = [-1.0, eps, 1.0]
    inits = [(re, im) for im in im_vals for re in re_vals]

    print("\n=== Lambda initialization grid (9 runs) ===")
    for (a, b) in inits:
        print(f"  init λ = ({a:+.3f}, {b:+.3f}i)")

    results = []
    for k, (a, b) in enumerate(inits, start=1):
        run_dir = base / f"init_lamR_{a:+.3f}_lamI_{b:+.3f}"
        print(f"\n=== RUN {k}/{len(inits)} init λ=({a:+.3f},{b:+.3f}i) ===")
        res = train_one_run(a, b, run_dir, seed_offset=0)
        results.append(res)
        print(f"--> converged λ ≈ {res['final_lamR']:.6f} + {res['final_lamI']:.6f} i, |λ|={res['final_abs_lam']:.3e}")

    # Save CSV summary
    csv_path = base / "grid_results_9pts_-1to1_eps_eps.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)

    # Print sorted by |λ|
    results_sorted = sorted(results, key=lambda d: d["final_abs_lam"])
    print("\n=== Sorted by |λ_final| (smallest first) ===")
    for r in results_sorted:
        print(
            f"init=({r['init_lamR']:+.3f},{r['init_lamI']:+.3f}i)  "
            f"-> final=({r['final_lamR']:+.6f},{r['final_lamI']:+.6f}i)  "
            f"|λ|={r['final_abs_lam']:.3e}  loss={r['final_loss']:.3e}"
        )

    print(f"\nSaved sweep summary CSV:\n  {csv_path}")
    print(f"\nSaved per-run figures under:\n  {base}")

if __name__ == "__main__":
    main()
