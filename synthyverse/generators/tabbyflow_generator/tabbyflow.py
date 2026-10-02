from typing import Literal
from copy import deepcopy
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..base import BaseGenerator
from .flow_model import ExpVFM
from ..tabdiff_generator.modules import UniModMLP

from ..dgm_utils import (
    FastTensorDataLoader,
    QuantileStandardScaler,
    clone_state_dict,
    split_validation,
    validate_c2st,
)
from ...utils.utils import resolve_epochs_from_training_steps


LRScheduler = Literal["reduce_lr_on_plateau", "anneal", "fixed"]
CLossWeightSchedule = Literal["anneal", "fixed"]


class TabbyFlowGenerator(BaseGenerator):
    def __init__(
        self,
        epochs: int = 8000,
        training_steps: int | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0,
        batch_size: int = 4096,
        ema_decay: float = 0.997,
        lr_scheduler: LRScheduler = "reduce_lr_on_plateau",
        reduce_lr_patience: int = 50,
        factor: float = 0.90,
        closs_weight_schedule: CLossWeightSchedule = "anneal",
        c_lambda: float = 1.0,
        d_lambda: float = 1.0,
        num_layers: int = 2,
        d_token: int = 4,
        n_head: int = 1,
        mlp_factor: int = 32,
        bias: bool = True,
        embedding_dim: int = 1024,
        mlp_dim: int = 2048,
        mlp_layers: int = 2,
        max_grad_norm: float = 1.0,
        warmup_epochs: int = 100,
        val_size: float = -1,
        val_steps: int = -1,
        patience: int = 3,
        max_validation_rows: int = 30_000,
        random_state: int = 0,
        full_determinism: bool = False
        ):
        super().__init__(random_state, full_determinism)

        self.epochs = epochs
        self.training_steps = training_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.ema_decay = ema_decay
        self.lr_scheduler = lr_scheduler
        self.reduce_lr_patience = reduce_lr_patience
        self.factor = factor
        self.closs_weight_schedule = closs_weight_schedule
        self.c_lambda = c_lambda
        self.d_lambda = d_lambda
        self.num_layers = num_layers
        self.d_token = d_token
        self.n_head = n_head
        self.mlp_factor = mlp_factor
        self.bias = bias
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.mlp_layers = mlp_layers
        self.max_grad_norm = max_grad_norm
        self.warmup_epochs = warmup_epochs
        self.val_size = val_size
        self.val_steps = val_steps
        self.patience = patience
        self.max_validation_rows = max_validation_rows
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit(self, X, discrete_features):

        self.discrete_features = discrete_features
        self.numerical_features = [col for col in X.columns if col not in discrete_features]
        self.col_order = X.columns
        X_train = X.copy()
        self.quant_encoder = QuantileStandardScaler(X_train.shape[0], self.random_state)
        X_train[self.numerical_features] = self.quant_encoder.fit_transform(
            X_train[self.numerical_features].astype(float)
        )

        X_discrete = torch.tensor(X_train[self.discrete_features].to_numpy())
        X_numerical = torch.tensor(X_train[self.numerical_features].to_numpy())
        X_train = torch.cat((X_numerical, X_discrete), dim=1).float()
        self.d_numerical = X_numerical.shape[1]
        self.categories = (
            np.array(
                self._categorical_cardinalities(self.discrete_features),
                dtype=np.int64,
            )
            if self.discrete_features
            else np.array([], dtype=np.int64)
        )
        train_loader = FastTensorDataLoader(X_train, batch_size=self.batch_size, shuffle=True)

        self.epochs = resolve_epochs_from_training_steps(
            self.epochs,
            self.training_steps,
            len(X),
            self.batch_size,
        )
        backbone = UniModMLP(
            d_numerical=self.d_numerical,
            categories=(self.categories).tolist(),
            num_layers=self.num_layers,
            d_token=self.d_token,
            n_head=self.n_head,
            factor=self.mlp_factor,
            bias=self.bias,
            embedding_dim=self.embedding_dim,
            mlp_dim=self.mlp_dim,
            mlp_layers=self.mlp_layers,
        )
        backbone.to(self.device)
        self.flow = ExpVFM(
                num_classes=self.categories,
                num_numerical_features=len(self.numerical_features),
                vf_fn=backbone,
                device=self.device,
        )
        self.flow.to(self.device)
        self.flow.train()
        ema_model = deepcopy(self.flow._vf_fn)
        for param in ema_model.parameters():
            param.detach_()
        best_ema_model = None

        optimizer = torch.optim.AdamW(self.flow.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.factor, patience=self.reduce_lr_patience)

        def _anneal_lr(step):
            frac_done = step / self.epochs
            lr = self.lr * (1 - frac_done)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        def update_ema(target_params, source_params, rate=0.999):
            """
            Update target parameters to be closer to those of source parameters using
            an exponential moving average.
            :param target_params: the target parameter sequence.
            :param source_params: the source parameter sequence.
            :param rate: the EMA rate (closer to 1 means slower).
            """
            for target, source in zip(target_params, source_params):
                target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)

        def to_ema_model():
            curr_model = self.flow._vf_fn
            self.flow._vf_fn = ema_model  # temporarily install the ema parameters into the model

            return curr_model

        def to_model(curr_model):
            self.flow._vf_fn = curr_model      # give back the parameters


        def _run_step(x, closs_weight, dloss_weight):
            x = x.to(self.device)

            self.flow.train()

            optimizer.zero_grad()

            dloss, closs = self.flow.mixed_loss(x)

            loss = dloss_weight * dloss + closs_weight * closs
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.max_grad_norm)
            optimizer.step()

            return dloss, closs

        def compute_loss():
            curr_dloss = 0.0
            curr_closs = 0.0
            curr_count = 0
            data_iter = train_loader
            for batch in data_iter:
                x = batch[0].float().to(self.device)
                self.flow.eval()
                with torch.no_grad():
                    batch_dloss, batch_closs = self.flow.mixed_loss(x)
                curr_dloss += batch_dloss.item() * len(x)
                curr_closs += batch_closs.item() * len(x)
                curr_count += len(x)
            mloss = np.around(curr_dloss / curr_count, 4)
            gloss = np.around(curr_closs / curr_count, 4)
            return mloss, gloss


        curr_epoch = 0
        closs_weight, dloss_weight = self.c_lambda, self.d_lambda
        best_ema_loss = np.inf
        start_time = time.monotonic()

        for epoch in range(curr_epoch, self.epochs):
            curr_epoch = epoch+1
            # Set up pbar
            pbar = tqdm(train_loader, total=len(train_loader))
            pbar.set_description(f"Epoch {epoch+1}/{self.epochs}")

            # Compute the loss weights
            if self.closs_weight_schedule == "fixed":
                pass
            elif self.closs_weight_schedule == "anneal":
                frac_done = epoch / self.epochs
                closs_weight = self.c_lambda * (1 - frac_done)
            else:
                raise NotImplementedError(f"The continuous loss weight schedule {self.closs_weight_schedule} is not implemneted")

            # Training Step
            curr_dloss = 0.0
            curr_closs = 0.0
            curr_count = 0
            curr_lr = optimizer.param_groups[0]['lr']
            for batch in pbar:
                x = batch[0].float().to(self.device)
                batch_dloss, batch_closs = _run_step(x, closs_weight, dloss_weight)
                curr_dloss += batch_dloss.item() * len(x)
                curr_closs += batch_closs.item() * len(x)
                curr_count += len(x)
                pbar.set_postfix({
                    "lr": curr_lr,
                    "DLoss": np.around(curr_dloss/curr_count, 4),
                    "CLoss": np.around(curr_closs/curr_count, 4),
                    "TotalLoss": np.around((curr_dloss + curr_closs)/curr_count, 4),
                    "closs_weight": closs_weight,
                    "dloss_weight": dloss_weight,
                })

            # Log training Loss
            log_dict = {}
            mloss = np.around(curr_dloss / curr_count, 4)
            gloss = np.around(curr_closs / curr_count, 4)
            total_loss = mloss + gloss
            if np.isnan(gloss):
                    print('Finding Nan in gaussian loss')
                    break
            loss_dict = {
                "epoch": epoch + 1,
                "lr": curr_lr,
                "closs_weight": closs_weight,
                "dloss_weight": dloss_weight,
                "loss/c_loss": gloss,
                "loss/d_loss": mloss,
                "loss/total_loss": total_loss
            }
            log_dict.update(loss_dict)

            # Adjust learning rate (warmup overrides during early epochs)
            if self.warmup_epochs > 0 and (epoch + 1) <= self.warmup_epochs:
                warmup_lr = self.lr * (epoch + 1) / self.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr
            elif self.lr_scheduler == 'reduce_lr_on_plateau':
                scheduler.step(total_loss)
            elif self.lr_scheduler == 'anneal':
                _anneal_lr(epoch)
            elif self.lr_scheduler == 'fixed':
                pass
            else:
                raise NotImplementedError(f"LR scheduler with name '{self.lr_scheduler}' is not implemented")

            # Update EMA models
            update_ema(ema_model.parameters(), self.flow._vf_fn.parameters(), rate=self.ema_decay)

            # Compute and log EMA model loss
            curr_model = to_ema_model()
            ema_mloss, ema_gloss = compute_loss()
            to_model(curr_model)
            ema_total_loss = ema_mloss + ema_gloss

            # Save the best ema ckpt
            if ema_total_loss < best_ema_loss and curr_epoch > 4000:
                best_ema_loss = ema_total_loss
                best_ema_model = clone_state_dict(ema_model)

        if best_ema_model is None:
            self.flow._vf_fn.load_state_dict(ema_model.state_dict())
        else:
            self.flow._vf_fn.load_state_dict(best_ema_model)
        self.flow.eval()
        train_time = time.monotonic() - start_time
        return self


    def _generate(self, n):
        self.flow.eval()
        with torch.no_grad():
            syn_X = self.flow.sample_all(n, self.batch_size, keep_nan_samples=True)

        syn_X_num, syn_X_cat = syn_X[:, :len(self.numerical_features)], syn_X[:, len(self.numerical_features):]
        syn_X_discrete = syn_X_cat.long().numpy()
        syn_X_numerical = syn_X_num.numpy()
        syn_X_numerical = self.quant_encoder.inverse_transform(syn_X_numerical)

        syn_X = pd.concat((pd.DataFrame(syn_X_discrete), pd.DataFrame(syn_X_numerical)), axis=1)
        syn_X.columns = self.discrete_features + self.numerical_features
        syn_X = syn_X[self.col_order]
        return syn_X

    def _save_extra(self, path):
        return super()._save_extra(path)

    def _load_extra(self, path):
        return super()._load_extra(path)



