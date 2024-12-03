import os
import math
import warnings
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import pandas as pd
import numpy as np

from typing import Callable, List, Tuple

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    # length_min: float = 0.5**9
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
    ):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach() # 从GP模型中提取长度尺度参数
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next

def bo_turbo(batch_size : int, num_dimensions : int, n_init : int,
             input_file: str, results_file: str, func_eval_batch : Callable, n_max_eval : int = 10000):
        
    max_cholesky_size = float("inf") # Always use Cholesky?
    torch.manual_seed(0)
    
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4
    N_CANDIDATES = min(5000, max(2000, 200 * num_dimensions)) if not SMOKE_TEST else 4

    if input_file and os.path.exists(input_file):
        input_rows = pd.read_csv(input_file)
        print(f'{input_rows['x_0'].count()} particles are provided with initial position')
        print(f'{input_rows['score'].count()} of them are provided with initial score')
        n_init = n_init - input_rows['x_0'].count()
        lack_score = input_rows['score'].isna()
        temp_swarm = input_rows.loc[lack_score, [f'x_{i}' for i in range(num_dimensions)]].to_numpy()
        temp_scores = func_eval_batch(temp_swarm)
        input_rows.loc[lack_score, 'score'] = temp_scores

        X_input = torch.tensor(
            input_rows[[f'x_{i}' for i in range(num_dimensions)]].to_numpy(), dtype=dtype, device=device
            )
        Y_input = torch.tensor(
            input_rows['score'].to_numpy(), dtype=dtype, device=device
            ).unsqueeze(-1)
    else:
        print('No initial position provided, initial position from Sobol series adopted')

    if n_init > 0:
        X_turbo = get_initial_points(num_dimensions, n_init)
        Y_turbo = torch.tensor(
            func_eval_batch(X_turbo.tolist()), dtype=dtype, device=device
        ).unsqueeze(-1)
        if input_file and os.path.exists(input_file):
            # Append data
            X_turbo = torch.cat((X_input, X_turbo), dim=0)
            Y_turbo = torch.cat((Y_input, Y_turbo), dim=0)
    else:
        X_turbo = X_input
        Y_turbo = Y_input
    
    result_df = pd.DataFrame(X_turbo.cpu(), columns=[f'x_{i}' for i in range(num_dimensions)])
    result_df.insert(0, 'score', Y_turbo.cpu())
    result_df.to_csv(results_file, float_format='%.5e', index=False)
    n_eval  = result_df.shape[0]
    
    state = TurboState(num_dimensions, batch_size=batch_size, best_value=max(Y_turbo).item()) # change
    while (n_eval < n_max_eval) and (not state.restart_triggered):  # Run until TuRBO converges
        # Fit a GP model
        train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, ard_num_dims=num_dimensions, lengthscale_constraint=Interval(0.005, 4.0)
            )
        )
        model = SingleTaskGP(
            X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
        # Do the fitting and acquisition function optimization inside the Cholesky context
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            fit_gpytorch_mll(mll)
    
            # Create a batch
            X_next = generate_batch(
                state=state,
                model=model,
                X=X_turbo,
                Y=train_Y,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                acqf="ts",
            )
    
        Y_next = torch.tensor(
            func_eval_batch(X_next.tolist()), dtype=dtype, device=device
        ).unsqueeze(-1)
    
        # Update state
        state = update_state(state=state, Y_next=Y_next)
    
        # Append data
        X_turbo = torch.cat((X_turbo, X_next), dim=0)
        Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

        result_df = pd.DataFrame(X_turbo.cpu(), columns=[f'x_{i}' for i in range(num_dimensions)])
        result_df['score'] = np.array(Y_turbo.cpu())
        result_df = result_df[['score'] + [f'x_{i}' for i in range(num_dimensions)]]
        result_df.to_csv(results_file, float_format='%.5e', index=False)
        n_eval  = result_df.shape[0]
    
        # Print current status
        print(
            f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
        )