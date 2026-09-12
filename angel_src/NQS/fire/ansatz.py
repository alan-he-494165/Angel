"""FiRE wave function (Sec. 4.2 of arXiv:2504.06087).

Equation map
------------
Eq. 15      :meth:`FiREWaveFunction.__call__`      Psi = J * sum_d det(Phi_d)
Eq. 16/17   :func:`pair_features`                  e_ij, e_im
Eq. 18      :class:`InitialEmbedding`              h^0 from nuclei
Eq. 19      :func:`rescaled_features`              log(1+r)/r scaling
Eq. 20/21   :class:`SpatialFilter`                 Gamma, chi, f_cut
Eq. 22/23   :class:`MessagePassing`                h^1 = h^0 + m_par + m_anti
Eq. 24      :class:`FiREWaveFunction`              h = MLP(h^1)
Eq. 25/26   :class:`Orbitals`                      Phi_dil and envelopes
Eq. 27      :class:`Jastrow`                       J = J_cusp + J_MLP + J_att
Eq. 28      :class:`Jastrow`                       analytic electron cusps
Eq. 29      :class:`Jastrow`                       per-electron MLP Jastrow
Eq. 30/31   :class:`RegisterAttention`             cross-attention Jastrow

Deviations from the paper, all documented in ``README.md``:

* neighborhoods N_{r_i} (Eq. 6) are evaluated as dense masked distance
  matrices rather than the padded ragged tensors of Sec. I, so the
  O(n_el) complexity reduction is not realised; values are identical.
* Eq. 25 carries no spin index.  Since h^0 itself is spin-independent, a
  single readout would make the rows of two coincident opposite-spin
  electrons identical.  The orbital readout weights are therefore
  spin-resolved (as in full-determinant FermiNet), while the embeddings
  and envelopes are shared.
* the paper does not state its activation function; tanh is used
  throughout because the Laplacian of the log-amplitude is needed.
"""

from __future__ import annotations

from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .config import AnsatzConfig
from .system import MolecularSystem

Array = jax.Array


def polynomial_cutoff(distance: Array, cutoff: float, order: int = 6) -> Array:
    """Smooth polynomial cutoff f_cut, one at r=0 and exactly zero at r>=c.

    The envelope of Gasteiger et al. used in Eq. 21; ``order`` is p.
    """

    x = jnp.clip(distance / cutoff, 0.0, 1.0)
    p = float(order)
    value = (
        1.0
        - 0.5 * (p + 1.0) * (p + 2.0) * x**p
        + p * (p + 2.0) * x ** (p + 1.0)
        - 0.5 * p * (p + 1.0) * x ** (p + 2.0)
    )
    return jnp.where(distance < cutoff, value, 0.0)


def safe_distance(displacement: Array, mask_self: Array | None = None) -> Array:
    """Euclidean norm with a gradient-safe treatment of coincident points."""

    squared = jnp.sum(displacement**2, axis=-1) + 1e-24
    if mask_self is not None:
        squared = jnp.where(mask_self, 1.0, squared)
    distance = jnp.sqrt(squared)
    if mask_self is not None:
        distance = jnp.where(mask_self, 0.0, distance)
    return distance


def pair_features(displacement: Array, distance: Array) -> Array:
    """Eq. 16/17: Concat[|r_i - r_j|, r_i - r_j]."""

    return jnp.concatenate([distance[..., None], displacement], axis=-1)


def rescaled_features(features: Array, distance: Array) -> Array:
    """Eq. 19: log(1 + r) / r scaling of a pair feature vector."""

    scale = jnp.where(distance > 1e-12, jnp.log1p(distance) / jnp.where(distance > 1e-12, distance, 1.0), 1.0)
    return features * scale[..., None]


class MLP(nn.Module):
    """Dense stack with tanh activations; the last layer is linear."""

    widths: Sequence[int]
    out_dim: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for width in self.widths:
            x = jnp.tanh(nn.Dense(width, use_bias=self.use_bias)(x))
        return nn.Dense(self.out_dim, use_bias=self.use_bias)(x)


class GatedLinearUnit(nn.Module):
    """LayerNorm followed by a gated linear unit (Eq. 18)."""

    out_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.LayerNorm()(x)
        value = nn.Dense(self.out_dim)(x)
        gate = nn.Dense(self.out_dim)(x)
        return value * jax.nn.sigmoid(gate)


class SpatialFilter(nn.Module):
    """Gamma of Eq. 20, with the radial basis chi of Eq. 21.

    ``n_centers`` > 1 gives each nucleus its own edge-MLP weights W_m, b_m
    (Gamma_m); the projection W, the Gaussian widths and W_env are shared.
    """

    out_dim: int
    cutoff: float
    edge_widths: Sequence[int] = (16, 8)
    n_gaussians: int = 32
    poly_order: int = 6
    n_centers: int = 1

    @nn.compact
    def __call__(self, edge: Array, distance: Array) -> Array:
        # edge: (..., n_centers, 4) when n_centers > 1 else (..., 4)
        x = edge
        in_dim = x.shape[-1]
        for width in self.edge_widths:
            if self.n_centers > 1:
                weight = self.param(
                    f"w_center_{in_dim}_{width}",
                    nn.initializers.lecun_normal(),
                    (self.n_centers, in_dim, width),
                )
                bias = self.param(
                    f"b_center_{in_dim}_{width}", nn.initializers.zeros, (self.n_centers, width)
                )
                x = jnp.einsum("...mi,mio->...mo", x, weight) + bias
            else:
                x = nn.Dense(width)(x)
            x = jnp.tanh(x)
            in_dim = width
        x = nn.Dense(self.out_dim, use_bias=False)(x)

        sigma = self.param(
            "gaussian_widths",
            lambda key, shape: jnp.asarray(
                np.linspace(0.5, 4.0, shape[0]) / max(self.cutoff, 1e-6)
            ),
            (self.n_gaussians,),
        )
        radial = jnp.exp(-((sigma * distance[..., None]) ** 2))
        radial = radial * polynomial_cutoff(distance, self.cutoff, self.poly_order)[..., None]
        radial = nn.Dense(self.out_dim, use_bias=False)(radial)
        return x * radial


class InitialEmbedding(nn.Module):
    """Eq. 18: nucleus-conditioned one-electron embedding h^0."""

    hidden_dim: int
    n_nuclei: int
    cutoff_nuclei: float
    edge_widths: Sequence[int]
    n_gaussians: int
    poly_order: int

    @nn.compact
    def __call__(self, edge_en: Array, edge_en_scaled: Array, distance_en: Array) -> Array:
        # edge_en: (n_el, n_nuc, 4)
        gamma = SpatialFilter(
            out_dim=self.hidden_dim,
            cutoff=self.cutoff_nuclei,
            edge_widths=self.edge_widths,
            n_gaussians=self.n_gaussians,
            poly_order=self.poly_order,
            n_centers=self.n_nuclei,
            name="gamma_nuc",
        )(edge_en, distance_en)
        nuc_embedding = self.param(
            "nucleus_embedding",
            nn.initializers.normal(stddev=1.0),
            (self.n_nuclei, self.hidden_dim),
        )
        projected = nn.Dense(self.hidden_dim, use_bias=False, name="edge_projection")(
            edge_en_scaled
        )
        message = jnp.sum(gamma * (nuc_embedding[None] + projected), axis=1)
        return GatedLinearUnit(self.hidden_dim, name="glu")(message)


class MessagePassing(nn.Module):
    """Eq. 22/23: one spin-resolved message-passing step."""

    hidden_dim: int
    cutoff: float
    edge_widths: Sequence[int]
    n_gaussians: int
    poly_order: int

    @nn.compact
    def __call__(
        self,
        h0: Array,
        edge_ee: Array,
        edge_ee_scaled: Array,
        distance_ee: Array,
        spin: Array,
    ) -> Array:
        n_el = h0.shape[0]
        pair_input = jnp.concatenate(
            [
                jnp.broadcast_to(h0[:, None, :], (n_el, n_el, self.hidden_dim)),
                jnp.broadcast_to(h0[None, :, :], (n_el, n_el, self.hidden_dim)),
                edge_ee_scaled,
            ],
            axis=-1,
        )
        same_spin = spin[:, None] == spin[None, :]
        not_self = ~jnp.eye(n_el, dtype=bool)
        within = (distance_ee < self.cutoff) & not_self

        h1 = h0
        for label, channel_mask in (("par", same_spin), ("anti", ~same_spin)):
            gamma = SpatialFilter(
                out_dim=self.hidden_dim,
                cutoff=self.cutoff,
                edge_widths=self.edge_widths,
                n_gaussians=self.n_gaussians,
                poly_order=self.poly_order,
                n_centers=1,
                name=f"gamma_{label}",
            )(edge_ee, distance_ee)
            value = jnp.tanh(nn.Dense(self.hidden_dim, name=f"message_{label}")(pair_input))
            mask = (channel_mask & within)[..., None]
            h1 = h1 + jnp.sum(jnp.where(mask, gamma * value, 0.0), axis=1)
        return h1


class Orbitals(nn.Module):
    """Eq. 25/26: orbital matrices with nucleus-appropriate envelopes."""

    n_determinants: int
    n_electrons: int
    n_up: int
    n_nuclei: int
    n_envelopes: int
    charges: Array
    ecp_mask: Array

    @nn.compact
    def __call__(self, h: Array, distance_en: Array) -> Array:
        hidden_dim = h.shape[-1]
        weight_up = self.param(
            "readout_up",
            nn.initializers.lecun_normal(),
            (self.n_determinants, hidden_dim, self.n_electrons),
        )
        weight_down = self.param(
            "readout_down",
            nn.initializers.lecun_normal(),
            (self.n_determinants, hidden_dim, self.n_electrons),
        )
        is_up = (jnp.arange(self.n_electrons) < self.n_up)[None, :, None]
        projected_up = jnp.einsum("ih,dho->dio", h, weight_up)
        projected_down = jnp.einsum("ih,dho->dio", h, weight_down)
        projected = jnp.where(is_up, projected_up, projected_down)

        pi = self.param(
            "envelope_weights",
            nn.initializers.normal(stddev=1.0),
            (self.n_electrons, self.n_nuclei, self.n_envelopes),
        )
        # Envelope decays are initialised at the nuclear charge times a
        # geometric spread, so that the tightest envelope already carries
        # roughly the correct electron-nucleus cusp exp(-Z r) for core
        # electrons.  Initialising them at 1 (the paper works with effective
        # core potentials, where that is adequate) leaves all-electron Li/Be
        # with a badly wrong core and heavy-tailed local energies.
        def _decay_init(_key, shape, dtype=jnp.float_):
            spread = jnp.geomspace(0.5, 2.0, shape[1]) if shape[1] > 1 else jnp.ones(1)
            target = jnp.asarray(self.charges, dtype=dtype)[:, None] * spread[None, :]
            return jnp.log(jnp.expm1(jnp.clip(target, 0.1, None)))

        raw_sigma = self.param(
            "envelope_decays",
            _decay_init,
            (self.n_nuclei, self.n_envelopes),
        )
        sigma = jax.nn.softplus(raw_sigma)
        # A bare Coulomb singularity requires an electron-nucleus cusp, which
        # the exponential exp(-sigma r) supplies.  An ECP removes that
        # singularity, so pseudopotential QMC convention instead requires a
        # smooth, cusp-free core; exp(-sigma r^2) has zero radial derivative
        # at its nucleus.  This distinction is not an equation from the FiRE
        # paper (arXiv:2504.06087), but a pseudopotential-QMC requirement.
        exp_decay = jnp.exp(-sigma[None] * distance_en[:, :, None])
        if np.any(np.asarray(self.ecp_mask)):
            gaussian_decay = jnp.exp(-sigma[None] * distance_en[:, :, None] ** 2)
            decay = jnp.where(self.ecp_mask[None, :, None], gaussian_decay, exp_decay)
        else:
            # Keep the established all-electron graph bit-for-bit identical;
            # even an unselected branch can alter compiled derivative traces.
            decay = exp_decay
        envelope = jnp.einsum("ime,ome->io", decay, pi)  # (n_el, n_orb)
        return projected * envelope[None]


class RegisterAttention(nn.Module):
    """Eq. 30: cross-attention from register queries to electron embeddings."""

    n_registers: int
    register_dim: int

    @nn.compact
    def __call__(self, h: Array) -> Array:
        hidden_dim = h.shape[-1]
        queries = self.param(
            "register_queries",
            nn.initializers.normal(stddev=1.0 / np.sqrt(hidden_dim)),
            (self.n_registers, hidden_dim),
        )
        values = self.param(
            "register_values",
            nn.initializers.lecun_normal(),
            (self.n_registers, hidden_dim, self.register_dim),
        )
        logits = jnp.einsum("ih,rh->ri", h, queries)
        attention = jax.nn.softmax(logits, axis=-1)
        projected = jnp.einsum("ih,rhv->riv", h, values)
        registers = jnp.einsum("ri,riv->rv", attention, projected)
        return registers.reshape(-1)


class Jastrow(nn.Module):
    """Eq. 27-31: the three-term Jastrow factor.

    Returns ``(exponents, prefactors)`` with three entries each; the factor
    is ``sum_k exp(a_k) * b_k``, kept in this split form so the caller can
    evaluate it stably in the log domain.
    """

    widths: Sequence[int]
    n_registers: int
    register_dim: int

    @nn.compact
    def __call__(self, h: Array, distance_ee: Array, spin: Array) -> tuple[Array, Array]:
        n_el = h.shape[0]
        # --- Eq. 28: analytic cusp Jastrow --------------------------------
        omega_par = self.param("omega_par", lambda key: jnp.asarray(-0.25))
        omega_anti = self.param("omega_anti", lambda key: jnp.asarray(-0.5))
        alpha_par = self.param("alpha_par", lambda key: jnp.asarray(1.0))
        alpha_anti = self.param("alpha_anti", lambda key: jnp.asarray(1.0))
        same_spin = spin[:, None] == spin[None, :]
        upper = jnp.triu(jnp.ones((n_el, n_el), dtype=bool), k=1)
        cusp_par = (omega_par * alpha_par**2) / (alpha_par + distance_ee)
        cusp_anti = (omega_anti * alpha_anti**2) / (alpha_anti + distance_ee)
        exponent_cusp = jnp.sum(
            jnp.where(upper & same_spin, cusp_par, 0.0)
        ) + jnp.sum(jnp.where(upper & ~same_spin, cusp_anti, 0.0))

        # --- Eq. 29: per-electron MLP Jastrow -----------------------------
        exponent_mlp = jnp.sum(MLP(self.widths, 1, name="mlp_log")(h))
        prefactor_mlp = jnp.sum(MLP(self.widths, 1, name="mlp_node")(h))

        # --- Eq. 30/31: register cross-attention Jastrow ------------------
        registers = RegisterAttention(
            n_registers=self.n_registers,
            register_dim=self.register_dim,
            name="attention",
        )(h)
        exponent_att = jnp.squeeze(MLP(self.widths, 1, name="att_log")(registers))
        prefactor_att = jnp.squeeze(MLP(self.widths, 1, name="att_node")(registers))

        exponents = jnp.stack([exponent_cusp, exponent_mlp, exponent_att])
        prefactors = jnp.stack([jnp.ones(()), prefactor_mlp, prefactor_att])
        return exponents, prefactors


class FiREWaveFunction(nn.Module):
    """The full FiRE ansatz, evaluated for a single electron configuration."""

    config: AnsatzConfig
    nuclei: Array
    charges: Array
    n_up: int
    n_down: int
    ecp_mask: Array

    @property
    def n_electrons(self) -> int:
        return self.n_up + self.n_down

    @nn.compact
    def __call__(self, r: Array) -> tuple[Array, Array]:
        """Return ``(sign, log|Psi|)`` for ``r`` of shape (n_el, 3)."""

        cfg = self.config
        n_el = self.n_electrons
        nuclei = jnp.asarray(self.nuclei)
        spin = jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_down)])

        # --- pair features (Eq. 16/17, 19) --------------------------------
        displacement_ee = r[:, None, :] - r[None, :, :]
        self_mask = jnp.eye(n_el, dtype=bool)
        distance_ee = safe_distance(displacement_ee, self_mask)
        edge_ee = pair_features(displacement_ee, distance_ee)
        edge_ee_scaled = rescaled_features(edge_ee, distance_ee)

        displacement_en = r[:, None, :] - nuclei[None, :, :]
        distance_en = safe_distance(displacement_en)
        edge_en = pair_features(displacement_en, distance_en)
        edge_en_scaled = rescaled_features(edge_en, distance_en)

        # --- embeddings (Eq. 18, 22-24) -----------------------------------
        h0 = InitialEmbedding(
            hidden_dim=cfg.hidden_dim,
            n_nuclei=nuclei.shape[0],
            cutoff_nuclei=cfg.cutoff_nuclei,
            edge_widths=cfg.edge_mlp_widths,
            n_gaussians=cfg.n_gaussians,
            poly_order=cfg.cutoff_poly_order,
            name="initial_embedding",
        )(edge_en, edge_en_scaled, distance_en)
        h1 = MessagePassing(
            hidden_dim=cfg.hidden_dim,
            cutoff=cfg.cutoff,
            edge_widths=cfg.edge_mlp_widths,
            n_gaussians=cfg.n_gaussians,
            poly_order=cfg.cutoff_poly_order,
            name="message_passing",
        )(h0, edge_ee, edge_ee_scaled, distance_ee, spin)
        h = MLP(cfg.embedding_mlp_widths, cfg.hidden_dim, name="embedding_mlp")(h1)

        # --- determinants (Eq. 15, 25, 26) --------------------------------
        orbitals = Orbitals(
            n_determinants=cfg.n_determinants,
            n_electrons=n_el,
            n_up=self.n_up,
            n_nuclei=nuclei.shape[0],
            n_envelopes=cfg.n_envelopes,
            charges=self.charges,
            ecp_mask=self.ecp_mask,
            name="orbitals",
        )(h, distance_en)
        # Exposed for Hartree-Fock orbital pretraining (see pretrain.py).
        self.sow("intermediates", "orbital_matrices", orbitals)
        det_signs, det_logs = jnp.linalg.slogdet(orbitals)
        max_log = jnp.max(det_logs)
        det_sum = jnp.sum(det_signs * jnp.exp(det_logs - max_log))
        sign_det = jnp.sign(det_sum)
        log_det = max_log + jnp.log(jnp.abs(det_sum) + 1e-300)

        # --- Jastrow (Eq. 27-31) ------------------------------------------
        exponents, prefactors = Jastrow(
            widths=cfg.jastrow_mlp_widths,
            n_registers=cfg.n_registers,
            register_dim=cfg.register_dim,
            name="jastrow",
        )(h, distance_ee, spin)
        max_exponent = jnp.max(exponents)
        jastrow_sum = jnp.sum(jnp.exp(exponents - max_exponent) * prefactors)
        sign_jastrow = jnp.sign(jastrow_sum)
        log_jastrow = max_exponent + jnp.log(jnp.abs(jastrow_sum) + 1e-300)

        return sign_det * sign_jastrow, log_det + log_jastrow

def orbital_matrices(model: "FiREWaveFunction", params, r: Array) -> Array:
    """Return the ``(n_det, n_el, n_el)`` orbital matrices Phi_d for one sample."""

    _, state = model.apply(params, r, mutable=["intermediates"])
    return state["intermediates"]["orbital_matrices"][0]


def make_log_psi(
    system: MolecularSystem, config: AnsatzConfig
) -> tuple[FiREWaveFunction, Callable[..., Array], Callable[..., tuple[Array, Array]]]:
    """Build the model and convenience ``log|Psi|`` / ``(sign, log|Psi|)`` callables.

    Both callables take ``(params, r)`` with ``r`` of shape ``(n_el, 3)``;
    batch them with :func:`jax.vmap`.
    """

    ecp_mask = jnp.asarray(
        [False] * system.n_nuclei
        if system.ecp is None
        else [entry is not None for entry in system.ecp],
        dtype=bool,
    )
    model = FiREWaveFunction(
        config=config,
        nuclei=jnp.asarray(system.nuclei),
        charges=jnp.asarray(system.charges),
        n_up=system.n_up,
        n_down=system.n_down,
        ecp_mask=ecp_mask,
    )

    def sign_log_psi(params, r: Array) -> tuple[Array, Array]:
        return model.apply(params, r)

    def log_psi(params, r: Array) -> Array:
        return model.apply(params, r)[1]

    return model, log_psi, sign_log_psi


def init_params(model: FiREWaveFunction, key: Array, n_electrons: int):
    """Initialize parameters from a random electron configuration."""

    key, subkey = jax.random.split(key)
    r = jax.random.normal(subkey, (n_electrons, 3))
    return model.init(key, r)
