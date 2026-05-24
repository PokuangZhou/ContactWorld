import torch
import torch.nn as nn
import torch.nn.functional as F


class SquareLossSeq(nn.Module):
    """
    Simple squared error prediction loss.
    Works for [B, D, T, 1, 1] tensors or compatible broadcast shapes.
    """
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer (single-GPU version).
    Expects input of shape [T, B, D].
    """
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: [T, B, D]
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12))

        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class _BaseRegularizer(nn.Module):
    def __init__(
        self,
        projector=None,
        idm=None,
        idm_after_proj=True,
        sim_t_after_proj=True,
        eps=1e-4,
    ):
        super().__init__()
        self.projector = projector
        self.idm = idm
        self.idm_after_proj = idm_after_proj
        self.sim_t_after_proj = sim_t_after_proj
        self.eps = eps

    def _flatten_state(self, state):
        # state: [B, D, T, 1, 1] -> [B*T, D]
        b, d, t, h, w = state.shape
        x = state.squeeze(-1).squeeze(-1).transpose(1, 2).reshape(b * t, d)
        return x

    def _project_if_needed(self, state):
        if self.projector is None:
            return state
        b, d, t, h, w = state.shape
        x = self._flatten_state(state)              # [B*T, D]
        x = self.projector(x)                       # [B*T, Dp]
        dp = x.shape[-1]
        x = x.reshape(b, t, dp).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return x

    def _sim_t_loss(self, state):
        # state: [B, D, T, 1, 1]
        x = state.squeeze(-1).squeeze(-1).transpose(1, 2)  # [B, T, D]
        if x.size(1) < 2:
            return x.new_tensor(0.0)
        return F.mse_loss(x[:, 1:], x[:, :-1])

    def _idm_loss(self, state, actions):
        if self.idm is None or actions is None:
            return state.new_tensor(0.0)

        # state: [B,D,T,1,1], actions: [B,A,T]
        x = state.squeeze(-1).squeeze(-1).transpose(1, 2)  # [B, T, D]
        if x.size(1) < 2:
            return x.new_tensor(0.0)

        s_t = x[:, :-1].reshape(-1, x.size(-1))
        s_tp1 = x[:, 1:].reshape(-1, x.size(-1))

        a = actions.transpose(1, 2)[:, :-1].reshape(-1, actions.size(1))
        a_pred = self.idm(s_t, s_tp1)
        return F.mse_loss(a_pred, a)


class VC_IDM_Sim_Regularizer(_BaseRegularizer):
    """
    EB-JEPA style variance/covariance regularizer
    + optional temporal similarity
    + optional inverse dynamics
    """
    def __init__(
        self,
        cov_coeff=1.0,
        std_coeff=1.0,
        sim_coeff_t=0.0,
        idm_coeff=0.0,
        idm=None,
        projector=None,
        spatial_as_samples=False,
        idm_after_proj=True,
        sim_t_after_proj=True,
        eps=1e-4,
    ):
        super().__init__(
            projector=projector,
            idm=idm,
            idm_after_proj=idm_after_proj,
            sim_t_after_proj=sim_t_after_proj,
            eps=eps,
        )
        self.cov_coeff = cov_coeff
        self.std_coeff = std_coeff
        self.sim_coeff_t = sim_coeff_t
        self.idm_coeff = idm_coeff
        self.spatial_as_samples = spatial_as_samples

    def _std_loss(self, z):
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return torch.mean(F.relu(1.0 - std))

    def _cov_loss(self, z):
        z = z - z.mean(dim=0, keepdim=True)
        n, d = z.shape
        cov = (z.T @ z) / max(n - 1, 1)
        off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
        return (off_diag ** 2).mean()

    def forward(self, state, actions=None):
        state_for_vc = self._project_if_needed(state)
        flat = self._flatten_state(state_for_vc)

        std_loss = self._std_loss(flat)
        cov_loss = self._cov_loss(flat)

        sim_state = state_for_vc if self.sim_t_after_proj else state
        sim_t_loss = self._sim_t_loss(sim_state) if self.sim_coeff_t > 0 else flat.new_tensor(0.0)

        idm_state = state_for_vc if self.idm_after_proj else state
        idm_loss = self._idm_loss(idm_state, actions) if self.idm_coeff > 0 else flat.new_tensor(0.0)

        total = (
            self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
            + self.sim_coeff_t * sim_t_loss
            + self.idm_coeff * idm_loss
        )

        reg_dict = {
            "std_loss": float(std_loss.detach().cpu()),
            "cov_loss": float(cov_loss.detach().cpu()),
            "sim_t_loss": float(sim_t_loss.detach().cpu()),
            "idm_loss": float(idm_loss.detach().cpu()),
        }
        return total, total.detach(), reg_dict


class SIGReg_IDM_Sim_Regularizer(_BaseRegularizer):
    """
    Le-WM style SIGReg + optional temporal similarity + optional IDM.
    """
    def __init__(
        self,
        sigreg_coeff=1.0,
        sim_coeff_t=0.0,
        idm_coeff=0.0,
        sigreg=None,
        idm=None,
        projector=None,
        idm_after_proj=True,
        sim_t_after_proj=True,
        eps=1e-4,
    ):
        super().__init__(
            projector=projector,
            idm=idm,
            idm_after_proj=idm_after_proj,
            sim_t_after_proj=sim_t_after_proj,
            eps=eps,
        )
        self.sigreg_coeff = sigreg_coeff
        self.sim_coeff_t = sim_coeff_t
        self.idm_coeff = idm_coeff
        self.sigreg = sigreg if sigreg is not None else SIGReg()

    def _sigreg_loss(self, state):
        # state: [B, D, T, 1, 1] -> [T, B, D]
        x = state.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()  # [B, T, D]
        x = x.transpose(0, 1).contiguous()  # [T, B, D]
        return self.sigreg(x)

    def forward(self, state, actions=None):
        state_for_sig = self._project_if_needed(state)
        sigreg_loss = self._sigreg_loss(state_for_sig)

        sim_state = state_for_sig if self.sim_t_after_proj else state
        sim_t_loss = self._sim_t_loss(sim_state) if self.sim_coeff_t > 0 else sigreg_loss.new_tensor(0.0)

        idm_state = state_for_sig if self.idm_after_proj else state
        idm_loss = self._idm_loss(idm_state, actions) if self.idm_coeff > 0 else sigreg_loss.new_tensor(0.0)

        total = (
            self.sigreg_coeff * sigreg_loss
            + self.sim_coeff_t * sim_t_loss
            + self.idm_coeff * idm_loss
        )

        reg_dict = {
            "sigreg_loss": float(sigreg_loss.detach().cpu()),
            "sim_t_loss": float(sim_t_loss.detach().cpu()),
            "idm_loss": float(idm_loss.detach().cpu()),
        }
        return total, total.detach(), reg_dict