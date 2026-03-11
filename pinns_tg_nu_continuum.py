import math, random
import os, time, csv

import torch
import torch.nn as nn
import torch.optim as optim

# =================================================
# Config
# =================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_COLLOC_PDE  = 4096
N_COLLOC_NORM = 4096

LR = 2e-3
EPOCHS_ADAM = 25000
USE_LBFGS = True
LBFGS_MAX_ITER = 6000

SEED = 1

WIDTH = 128
DEPTH = 6
FOURIER_K = 1
DOMAIN = (0.0, 2*math.pi)

# Oversampling controls
USE_STAGNATION_OVERSAMPLE = True
FOCUS_FRAC = 0.15
FOCUS_SIGMA = 0.25

# Loss weights
W_PDE   = 10.0
W_NORM  = 5.0
W_PHASE = 1e-3
W_MEAN  = 1e-2
W_RQ    = 1.0
RQ_SCALE = 1e-3

# Initial lambda
INIT_LAMR = 1.0
INIT_LAMI = 1.0

# Viscosity sweep
NU_LIST = [0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.1]

SAVE_DIR = "out_phi_free_lambda_nu_sweep"
RUN_TAG = time.strftime("%Y%m%d_%H%M%S")


# =================================================
# Reproducibility
# =================================================
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =================================================
# Sampling
# =================================================
def sample_xy_uniform(n):
    x1 = torch.rand(n, 1, device=DEVICE) * (DOMAIN[1]-DOMAIN[0]) + DOMAIN[0]
    x2 = torch.rand(n, 1, device=DEVICE) * (DOMAIN[1]-DOMAIN[0]) + DOMAIN[0]
    return x1.requires_grad_(True), x2.requires_grad_(True)

def sample_xy_mixture(n, frac_focus=FOCUS_FRAC, sigma=FOCUS_SIGMA):
    if (not USE_STAGNATION_OVERSAMPLE) or frac_focus <= 0.0:
        return sample_xy_uniform(n)

    n_focus = int(frac_focus * n)
    n_uni = n - n_focus

    x1u = torch.rand(n_uni, 1, device=DEVICE) * (DOMAIN[1]-DOMAIN[0]) + DOMAIN[0]
    x2u = torch.rand(n_uni, 1, device=DEVICE) * (DOMAIN[1]-DOMAIN[0]) + DOMAIN[0]

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

        x1f = torch.remainder(xf[:, 0:1], 2*math.pi)
        x2f = torch.remainder(xf[:, 1:2], 2*math.pi)

        x1 = torch.cat([x1u, x1f], dim=0)
        x2 = torch.cat([x2u, x2f], dim=0)
    else:
        x1, x2 = x1u, x2u

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


# =================================================
# Autograd helpers
# =================================================
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


# =================================================
# Base fields (Taylor–Green stationary solution)
# =================================================
def us_omega_gradomega(x1, x2):
    us1 = -torch.cos(x1)*torch.sin(x2)
    us2 =  torch.sin(x1)*torch.cos(x2)
    omega_s = 2.0*torch.cos(x1)*torch.cos(x2)
    domg_dx1 = -2.0*torch.sin(x1)*torch.cos(x2)
    domg_dx2 = -2.0*torch.cos(x1)*torch.sin(x2)
    return (us1, us2), omega_s, (domg_dx1, domg_dx2)


# =================================================
# Network
# =================================================
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
    def __init__(self, K=1, width=128, depth=5, init_lamr=INIT_LAMR, init_lami=INIT_LAMI):
        super().__init__()
        self.model = MLP(4*K, 2, width, depth)
        self.lamR = nn.Parameter(torch.tensor(float(init_lamr), dtype=torch.float32))
        self.lamI = nn.Parameter(torch.tensor(float(init_lami), dtype=torch.float32))

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


# =================================================
# Loss (with viscosity)
# =================================================
def loss_on_points(model, nu, x1_pde, x2_pde, x1_norm, x2_norm):
    psi_R, psi_I = model(x1_pde, x2_pde)

    lap_psi_R = laplacian(psi_R, x1_pde, x2_pde)
    lap_psi_I = laplacian(psi_I, x1_pde, x2_pde)
    omg_R = -lap_psi_R
    omg_I = -lap_psi_I

    # Grad omega
    domgR_dx1 = grad(omg_R, x1_pde); domgR_dx2 = grad(omg_R, x2_pde)
    domgI_dx1 = grad(omg_I, x1_pde); domgI_dx2 = grad(omg_I, x2_pde)

    # u' from psi
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

    # Viscous term: -nu * Δω
    if nu != 0.0:
        lap_omg_R = laplacian(omg_R, x1_pde, x2_pde)
        lap_omg_I = laplacian(omg_I, x1_pde, x2_pde)
    else:
        lap_omg_R = torch.zeros_like(omg_R)
        lap_omg_I = torch.zeros_like(omg_I)

    Rw_R = (lamR*omg_R - lamI*omg_I) + us_dot_grad_omg_R + upr_dot_grad_omegas_R - nu * lap_omg_R
    Rw_I = (lamR*omg_I + lamI*omg_R) + us_dot_grad_omg_I + upr_dot_grad_omegas_I - nu * lap_omg_I
    L_pde = (Rw_R.pow(2) + Rw_I.pow(2)).mean()

    # Norm on uniform batch
    psi_Rn, psi_In = model(x1_norm, x2_norm)
    omg_Rn = -laplacian(psi_Rn, x1_norm, x2_norm)
    omg_In = -laplacian(psi_In, x1_norm, x2_norm)
    mean_omg2 = (omg_Rn.pow(2) + omg_In.pow(2)).mean()
    L_norm_omg = (mean_omg2 - 1.0)**2

    # Phase anchor
    x0 = torch.zeros((1,1), device=DEVICE, requires_grad=True)
    y0 = torch.zeros((1,1), device=DEVICE, requires_grad=True)
    psiR0, psiI0 = model(x0, y0)
    L_phase = psiI0.pow(2).mean()

    # Mean-zero gauge
    L_mean = psi_Rn.mean().pow(2) + psi_In.mean().pow(2)

    # Rayleigh quotient consistency (viscous-consistent rhs)
    rhsR = -(us_dot_grad_omg_R + upr_dot_grad_omegas_R - nu * lap_omg_R)
    rhsI = -(us_dot_grad_omg_I + upr_dot_grad_omegas_I - nu * lap_omg_I)

    numR = (omg_R*rhsR + omg_I*rhsI).mean()
    numI = (omg_R*rhsI - omg_I*rhsR).mean()
    den  = (omg_R.pow(2)+omg_I.pow(2)).mean() + 1e-12
    rqR, rqI = numR/den, numI/den
    L_rq = RQ_SCALE*((lamR - rqR).pow(2) + (lamI - rqI).pow(2))

    loss = W_PDE*L_pde + W_NORM*L_norm_omg + W_PHASE*L_phase + W_MEAN*L_mean + W_RQ*L_rq

    parts = dict(
        nu=float(nu),
        loss=float(loss.detach()),
        L_pde=float(L_pde.detach()),
        mean_omg2=float(mean_omg2.detach()),
        L_rq=float(L_rq.detach()),
        lamR=float(lamR.detach()),
        lamI=float(lamI.detach()),
        rqR=float(rqR.detach()),
        rqI=float(rqI.detach()),
    )
    return loss, parts

def make_batches():
    x1_pde,  x2_pde  = sample_xy_mixture(N_COLLOC_PDE)
    x1_norm, x2_norm = sample_xy_uniform(N_COLLOC_NORM)
    return x1_pde, x2_pde, x1_norm, x2_norm


# =================================================
# Training for one nu (optionally continuation)
# =================================================
def train_for_nu(nu, run_dir, prev_ckpt=None, init_lamr=INIT_LAMR, init_lami=INIT_LAMI, load_prev_weights=True):
    os.makedirs(run_dir, exist_ok=True)

    model = EigenPINN(K=FOURIER_K, width=WIDTH, depth=DEPTH,
                      init_lamr=init_lamr, init_lami=init_lami).to(DEVICE)

    if load_prev_weights and prev_ckpt is not None and os.path.exists(prev_ckpt):
        state = torch.load(prev_ckpt, map_location=DEVICE)
        model.load_state_dict(state["model_state"], strict=True)

    opt = optim.Adam(model.parameters(), lr=LR)

    for it in range(EPOCHS_ADAM):
        opt.zero_grad(set_to_none=True)
        x1_pde, x2_pde, x1_norm, x2_norm = make_batches()
        loss, parts = loss_on_points(model, nu, x1_pde, x2_pde, x1_norm, x2_norm)
        loss.backward()
        opt.step()

        if (it+1) % 10 == 0:
            print(
                f"[nu={nu:0.4f} | Adam {it+1}] loss={parts['loss']:.3e} "
                f"lam=({parts['lamR']:.6f},{parts['lamI']:+.6f}i) "
                f"RQ=({parts['rqR']:.6f},{parts['rqI']:+.6f}i) "
                f"L_rq={parts['L_rq']:.2e} mean|omg|^2={parts['mean_omg2']:.3e} PDE={parts['L_pde']:.2e}"
            )

    if USE_LBFGS:
        x1_pde_fix, x2_pde_fix, x1_norm_fix, x2_norm_fix = make_batches()
        x1_pde_fix  = x1_pde_fix.detach().requires_grad_(True)
        x2_pde_fix  = x2_pde_fix.detach().requires_grad_(True)
        x1_norm_fix = x1_norm_fix.detach().requires_grad_(True)
        x2_norm_fix = x2_norm_fix.detach().requires_grad_(True)

        opt_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=LBFGS_MAX_ITER,
            tolerance_grad=1e-7,
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe"
        )

        def closure():
            opt_lbfgs.zero_grad(set_to_none=True)
            loss, _ = loss_on_points(model, nu, x1_pde_fix, x2_pde_fix, x1_norm_fix, x2_norm_fix)
            loss.backward()
            return loss

        opt_lbfgs.step(closure)
        loss, parts = loss_on_points(model, nu, x1_pde_fix, x2_pde_fix, x1_norm_fix, x2_norm_fix)
        print(
            f"[nu={nu:0.4f} | LBFGS] final loss={parts['loss']:.3e} "
            f"lam=({parts['lamR']:.6f},{parts['lamI']:+.6f}i) "
            f"RQ=({parts['rqR']:.6f},{parts['rqI']:+.6f}i) PDE={parts['L_pde']:.2e}"
        )

    ckpt_path = os.path.join(run_dir, "model.pt")
    torch.save({
        "nu": float(nu),
        "model_state": model.state_dict(),
        "lamR": float(model.lambda_R.detach().cpu()),
        "lamI": float(model.lambda_I.detach().cpu()),
        "seed": SEED,
    }, ckpt_path)

    summary = {
        "nu": float(nu),
        "lamR": float(model.lambda_R.detach().cpu()),
        "lamI": float(model.lambda_I.detach().cpu()),
        "rqR": float(parts["rqR"]),
        "rqI": float(parts["rqI"]),
        "loss": float(parts["loss"]),
        "L_pde": float(parts["L_pde"]),
        "L_rq": float(parts["L_rq"]),
        "mean_omg2": float(parts["mean_omg2"]),
        "ckpt": ckpt_path,
        "run_dir": run_dir,
    }
    return summary


# =================================================
# Sweep driver + CSV
# =================================================
def write_csv_row(csv_path, row, fieldnames):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    sweep_dir = os.path.join(SAVE_DIR, f"sweep_{RUN_TAG}")
    os.makedirs(sweep_dir, exist_ok=True)

    csv_path = os.path.join(sweep_dir, "results.csv")
    fields = ["nu","lamR","lamI","rqR","rqI","loss","L_pde","L_rq","mean_omg2","ckpt","run_dir"]

    prev_ckpt = None
    init_lamr = INIT_LAMR
    init_lami = INIT_LAMI

    for nu in NU_LIST:
        run_dir = os.path.join(sweep_dir, f"nu_{nu:0.4f}")
        print("\n" + "="*90)
        print(f"Starting nu={nu:.4f} | continuation from {prev_ckpt}")

        summary = train_for_nu(
            nu=nu,
            run_dir=run_dir,
            prev_ckpt=prev_ckpt,
            init_lamr=init_lamr,
            init_lami=init_lami,
            load_prev_weights=True,
        )

        write_csv_row(csv_path, summary, fields)

        # update for next run
        init_lamr = summary["lamR"]
        init_lami = summary["lamI"]
        prev_ckpt = summary["ckpt"]

        # stop early if Re(λ) has crossed zero
        if summary["lamR"] < -1e-3:
             break

    print("\nSweep finished.")
    print(f"CSV saved at: {csv_path}")
    print(f"All runs saved under: {sweep_dir}")

if __name__ == "__main__":
    main()
