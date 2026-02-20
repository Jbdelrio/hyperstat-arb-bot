# tests/test_projection_neutral.py
from __future__ import annotations

import numpy as np

from hyperstat.core.math import neutralize_weights


def test_neutralize_no_constraints_is_identity():
    """
    Si on ne demande aucune neutralisation, la fonction doit renvoyer les poids inchangés.
    """
    w = {"A": 0.3, "B": -0.1, "C": 0.2}
    w2 = neutralize_weights(w, dollar_neutral=False, beta_neutral=False)
    assert w2.keys() == w.keys()
    for k in w:
        assert abs(w2[k] - w[k]) < 1e-12


def test_dollar_neutral_sum_zero():
    """
    Dollar-neutral => somme des poids = 0 (à epsilon numérique près).
    """
    w = {"A": 1.0, "B": -0.2, "C": 0.1}
    w2 = neutralize_weights(w, dollar_neutral=True, beta_neutral=False)
    assert abs(sum(w2.values())) < 1e-9


def test_beta_neutral_sum_and_beta_zero():
    """
    Beta-neutral + dollar-neutral :
      - somme(w) = 0
      - somme(w_i * beta_i) = 0
    """
    w = {"A": 1.0, "B": -0.2, "C": 0.1}
    betas = {"A": 1.2, "B": 0.8, "C": 0.5}

    w2 = neutralize_weights(w, betas=betas, dollar_neutral=True, beta_neutral=True)

    # contrainte dollar-neutral
    assert abs(sum(w2.values())) < 1e-9

    # contrainte beta-neutral
    beta_exposure = sum(w2[k] * betas[k] for k in w2.keys())
    assert abs(beta_exposure) < 1e-6


def test_beta_neutral_handles_missing_betas_gracefully():
    """
    Si betas ne contient pas tous les symboles, on doit rester robuste (beta manquant -> 0).
    """
    w = {"A": 1.0, "B": -0.3, "C": 0.2}
    betas = {"A": 1.1, "B": 0.9}  # C manquant

    w2 = neutralize_weights(w, betas=betas, dollar_neutral=True, beta_neutral=True)

    assert abs(sum(w2.values())) < 1e-9
    beta_exposure = sum(w2[k] * float(betas.get(k, 0.0)) for k in w2.keys())
    assert abs(beta_exposure) < 1e-6


def test_neutralization_keeps_vector_in_span_reasonable():
    """
    Sanity check: la neutralisation ne doit pas exploser les poids.
    (Ce n'est pas une garantie math universelle, mais utile pour détecter bugs numériques.)
    """
    w = {"A": 0.5, "B": -0.4, "C": 0.2, "D": -0.1}
    betas = {"A": 1.3, "B": 0.7, "C": 1.0, "D": 0.2}

    w2 = neutralize_weights(w, betas=betas, dollar_neutral=True, beta_neutral=True)

    norm_in = float(np.linalg.norm(np.array(list(w.values()), dtype=float)))
    norm_out = float(np.linalg.norm(np.array(list(w2.values()), dtype=float)))

    # Le vecteur projeté peut être plus petit (projection), mais ne devrait pas exploser
    assert norm_out <= 10.0 * max(1e-12, norm_in)
