"""
src/hyperstat/core/credentials.py

Gestion sécurisée de la clé privée Hyperliquid.

Priorité de résolution (la première trouvée gagne) :
  1. Argument CLI  --hl-private-key 0x...
  2. Variable d'environnement  HL_PRIVATE_KEY
  3. Fichier .env dans le répertoire courant
  4. Session Streamlit st.session_state["hl_private_key"]
     (injecté par apps/dashboard.py via le widget de connexion)

Usage backtest / CLI :
    from hyperstat.core.credentials import resolve_credentials
    creds = resolve_credentials()
    print(creds.address, creds.private_key[:6] + "***")

Usage Streamlit :
    # Dans apps/dashboard.py, appeler render_credentials_sidebar()
    # qui stocke les valeurs dans st.session_state et retourne True si prêt.
    from hyperstat.core.credentials import render_credentials_sidebar
    ready = render_credentials_sidebar()
    if ready:
        creds = resolve_credentials(streamlit_session=st.session_state)
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─────────────────────────── Dataclass ────────────────────────────────

@dataclass
class HyperliquidCredentials:
    address: str          # 0x...  wallet address
    private_key: str      # 0x...  private key (EIP-712 signing)
    testnet: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validation minimale du format des clés."""
        addr_ok = bool(re.match(r"^0x[0-9a-fA-F]{40}$", self.address))
        key_ok  = bool(re.match(r"^(0x)?[0-9a-fA-F]{64}$", self.private_key))
        if not addr_ok:
            raise ValueError(
                f"HL_ADDRESS invalide : doit être une adresse Ethereum "
                f"(0x + 40 hex). Reçu : {self.address[:10]}..."
            )
        if not key_ok:
            raise ValueError(
                "HL_PRIVATE_KEY invalide : doit être 64 hex chars (optionnel 0x prefix)."
            )
        # Normalise : toujours avec préfixe 0x
        if not self.private_key.startswith("0x"):
            self.private_key = "0x" + self.private_key

    def masked(self) -> str:
        """Représentation sûre pour les logs."""
        return f"address={self.address}, key={self.private_key[:6]}...{self.private_key[-4:]}"


# ─────────────────────────── Résolution ───────────────────────────────

def resolve_credentials(
    cli_args: argparse.Namespace | None = None,
    streamlit_session: dict[str, Any] | None = None,
    dotenv_path: str | Path | None = None,
) -> HyperliquidCredentials:
    """
    Résout les credentials dans l'ordre de priorité défini ci-dessus.

    Args:
        cli_args           : namespace argparse (si appelé depuis le CLI).
        streamlit_session  : st.session_state (si appelé depuis Streamlit).
        dotenv_path        : chemin vers un fichier .env alternatif.

    Returns:
        HyperliquidCredentials validé.

    Raises:
        CredentialsMissingError si aucune source ne fournit les credentials.
    """
    # 1. CLI
    if cli_args is not None:
        addr = getattr(cli_args, "hl_address", None)
        key  = getattr(cli_args, "hl_private_key", None)
        testnet = getattr(cli_args, "testnet", False)
        if addr and key:
            return HyperliquidCredentials(address=addr, private_key=key, testnet=testnet)

    # 2. Variables d'environnement
    addr = os.environ.get("HL_ADDRESS", "")
    key  = os.environ.get("HL_PRIVATE_KEY", "")
    testnet = os.environ.get("HL_TESTNET", "false").lower() == "true"
    if addr and key:
        return HyperliquidCredentials(address=addr, private_key=key, testnet=testnet)

    # 3. Fichier .env
    _load_dotenv(dotenv_path or Path(".env"))
    addr = os.environ.get("HL_ADDRESS", "")
    key  = os.environ.get("HL_PRIVATE_KEY", "")
    testnet = os.environ.get("HL_TESTNET", "false").lower() == "true"
    if addr and key:
        return HyperliquidCredentials(address=addr, private_key=key, testnet=testnet)

    # 4. Session Streamlit
    if streamlit_session is not None:
        addr = streamlit_session.get("hl_address", "")
        key  = streamlit_session.get("hl_private_key", "")
        testnet = streamlit_session.get("hl_testnet", False)
        if addr and key:
            return HyperliquidCredentials(address=addr, private_key=key, testnet=testnet)

    raise CredentialsMissingError(
        "Impossible de résoudre les credentials Hyperliquid.\n"
        "Solutions :\n"
        "  • Export : export HL_ADDRESS=0x... HL_PRIVATE_KEY=0x...\n"
        "  • Fichier .env à la racine du projet\n"
        "  • CLI    : python -m hyperstat.main --hl-address 0x... --hl-private-key 0x...\n"
        "  • Streamlit : utiliser le panneau de connexion dans la sidebar"
    )


def add_credential_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Ajoute les arguments --hl-address et --hl-private-key à un parser argparse existant.

    Usage :
        parser = argparse.ArgumentParser()
        add_credential_args(parser)
        # Dans hyperstat/cli/commands.py, ajouter cet appel sur le sous-parser 'live'
    """
    grp = parser.add_argument_group("Hyperliquid credentials")
    grp.add_argument(
        "--hl-address",
        metavar="0x...",
        default=None,
        help="Adresse wallet Hyperliquid (optionnel si HL_ADDRESS est défini)",
    )
    grp.add_argument(
        "--hl-private-key",
        metavar="0x...",
        default=None,
        dest="hl_private_key",
        help=(
            "Clé privée EIP-712 (optionnel si HL_PRIVATE_KEY est défini). "
            "⚠ Ne jamais committer cette valeur dans le code source."
        ),
    )
    grp.add_argument(
        "--testnet",
        action="store_true",
        default=False,
        help="Utiliser le testnet Hyperliquid (sandbox)",
    )
    return parser


# ─────────────────────────── Streamlit widget ─────────────────────────

def render_credentials_sidebar() -> bool:
    """
    Affiche un panneau de connexion dans la sidebar Streamlit.
    Stocke les valeurs dans st.session_state et retourne True si les
    credentials sont présents et valides.

    Usage dans apps/dashboard.py :
        from hyperstat.core.credentials import render_credentials_sidebar, resolve_credentials
        import streamlit as st

        ready = render_credentials_sidebar()
        if not ready:
            st.stop()
        creds = resolve_credentials(streamlit_session=st.session_state)
        # ... reste du dashboard
    """
    try:
        import streamlit as st
    except ImportError:
        raise ImportError("Streamlit n'est pas installé. pip install streamlit")

    with st.sidebar:
        st.markdown("## 🔐 Connexion Hyperliquid")

        # Afficher le statut courant
        already_set = bool(
            st.session_state.get("hl_address")
            and st.session_state.get("hl_private_key")
        )

        if already_set:
            creds_display = (
                f"`{st.session_state['hl_address'][:8]}...`  "
                f"clé `{st.session_state['hl_private_key'][:8]}***`"
            )
            st.success(f"✅ Connecté — {creds_display}")
            if st.button("🔓 Déconnecter", use_container_width=True):
                st.session_state.pop("hl_address", None)
                st.session_state.pop("hl_private_key", None)
                st.session_state.pop("hl_testnet", None)
                st.rerun()
            return True

        with st.expander("➕ Entrer les credentials", expanded=True):
            st.caption(
                "Les clés ne sont **jamais** stockées sur disque — "
                "elles restent en mémoire pour la durée de la session uniquement."
            )

            addr = st.text_input(
                "Adresse wallet (HL_ADDRESS)",
                placeholder="0x1234...abcd",
                key="_input_hl_address",
            )
            key = st.text_input(
                "Clé privée (HL_PRIVATE_KEY)",
                placeholder="0xabcd...1234",
                type="password",   # masquée dans l'UI
                key="_input_hl_private_key",
            )
            testnet = st.checkbox("Utiliser le testnet", value=False)

            if st.button("🔒 Valider et connecter", use_container_width=True, type="primary"):
                if not addr or not key:
                    st.error("Adresse et clé privée requises.")
                    return False
                try:
                    # Validation format avant de stocker
                    HyperliquidCredentials(address=addr, private_key=key, testnet=testnet)
                    st.session_state["hl_address"]     = addr
                    st.session_state["hl_private_key"] = key
                    st.session_state["hl_testnet"]     = testnet
                    st.success("✅ Credentials validés.")
                    st.rerun()
                except ValueError as e:
                    st.error(f"Format invalide : {e}")
                    return False

        # Séparateur : charger depuis .env si présent
        st.markdown("---")
        st.caption("Ou charger depuis un fichier `.env` local :")
        uploaded = st.file_uploader(
            "Charger un fichier .env",
            type=["env", "txt"],
            key="_env_uploader",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            content = uploaded.read().decode("utf-8")
            parsed  = _parse_dotenv_string(content)
            addr    = parsed.get("HL_ADDRESS", "")
            key     = parsed.get("HL_PRIVATE_KEY", "")
            testnet = parsed.get("HL_TESTNET", "false").lower() == "true"
            if addr and key:
                try:
                    HyperliquidCredentials(address=addr, private_key=key, testnet=testnet)
                    st.session_state["hl_address"]     = addr
                    st.session_state["hl_private_key"] = key
                    st.session_state["hl_testnet"]     = testnet
                    st.success("✅ Chargé depuis .env.")
                    st.rerun()
                except ValueError as e:
                    st.error(f"Fichier .env invalide : {e}")
            else:
                st.warning("HL_ADDRESS ou HL_PRIVATE_KEY manquant dans le fichier.")

    return False


# ─────────────────────────── Helpers internes ─────────────────────────

def _load_dotenv(path: Path) -> None:
    """Charge un fichier .env dans os.environ sans dépendance externe."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:  # ne pas écraser une var déjà définie
            os.environ[k] = v


def _parse_dotenv_string(content: str) -> dict[str, str]:
    """Parse le contenu d'un fichier .env en dict."""
    result: dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        result[k.strip()] = v.strip().strip('"').strip("'")
    return result


# ─────────────────────────── Exception ────────────────────────────────

class CredentialsMissingError(Exception):
    """Levée quand aucune source ne fournit les credentials."""
