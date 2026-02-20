# src/hyperstat/main.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml
from rich import print as rprint

from hyperstat.core.logging import setup_logging
from hyperstat.core.clock import LiveClock
from hyperstat.core.risk import RiskState, KillSwitchConfig

app = typer.Typer(help="hyperstat - Hyperliquid stat-arb bot (backtest/paper/live)")


# -----------------------------
# Config utilities (deep merge)
# -----------------------------
def _read_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping (dict): {p}")
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base.
    - dict merges recursively
    - lists are replaced (not concatenated) by default
    - scalars replaced
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    default_path: str,
    env_path: Optional[str] = None,
    strategy_path: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = _read_yaml(default_path)
    if env_path:
        cfg = _deep_merge(cfg, _read_yaml(env_path))
    if strategy_path:
        cfg = _deep_merge(cfg, _read_yaml(strategy_path))
    return cfg


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _risk_cfg_from_config(cfg: Dict[str, Any]) -> KillSwitchConfig:
    risk = cfg.get("risk", {}) or {}
    return KillSwitchConfig(
        max_intraday_drawdown_pct=_as_float(risk.get("max_intraday_drawdown_pct"), 0.03),
        cooldown_minutes=_as_int(risk.get("cooldown_minutes"), 720),
        z_emergency_flat=_as_float(risk.get("z_emergency_flat"), 3.5),
    )


def _print_effective_config(cfg: Dict[str, Any]) -> None:
    # minimal, avoids leaking secrets (env vars only)
    view = {
        "run": cfg.get("run", {}),
        "exchange": {
            "name": cfg.get("exchange", {}).get("name"),
            "network": cfg.get("exchange", {}).get("network"),
            "rest_url": cfg.get("exchange", {}).get("rest_url"),
            "ws_url": cfg.get("exchange", {}).get("ws_url"),
            "limits": cfg.get("exchange", {}).get("limits", {}),
            "account": {
                "address_env": cfg.get("exchange", {}).get("account", {}).get("address_env"),
                "private_key_env": cfg.get("exchange", {}).get("account", {}).get("private_key_env"),
            },
        },
        "data": cfg.get("data", {}),
        "universe": cfg.get("universe", {}),
        "portfolio": cfg.get("portfolio", {}),
        "execution": cfg.get("execution", {}),
        "risk": cfg.get("risk", {}),
        "strategy": cfg.get("strategy", {}),
    }
    rprint("[bold cyan]Effective config (sanitized):[/bold cyan]")
    rprint(view)


def _common_bootstrap(config_files: List[str], log_level: Optional[str]) -> Dict[str, Any]:
    if not config_files:
        raise typer.BadParameter("Provide at least one config path (default.yaml).")

    default_path = config_files[0]
    env_path = config_files[1] if len(config_files) >= 2 else None
    strat_path = config_files[2] if len(config_files) >= 3 else None

    cfg = load_config(default_path, env_path, strat_path)

    # allow log override
    if log_level:
        cfg.setdefault("run", {})["log_level"] = log_level

    # setup logging
    lvl = (cfg.get("run", {}) or {}).get("log_level", "INFO")
    setup_logging(level=str(lvl))

    return cfg


# -----------------------------
# CLI Commands
# -----------------------------
@app.command("print-config")
def print_config(
    config: List[str] = typer.Option(
        ["configs/default.yaml", "configs/hyperliquid_testnet.yaml", "configs/strategy_stat_arb.yaml"],
        help="Config files: default, env override, strategy override (deep-merged).",
    ),
):
    cfg = _common_bootstrap(config, log_level=None)
    _print_effective_config(cfg)


@app.command("paper")
def paper(
    config: List[str] = typer.Option(
        ["configs/default.yaml", "configs/hyperliquid_testnet.yaml", "configs/strategy_stat_arb.yaml"],
        help="Config files: default, env override, strategy override (deep-merged).",
    ),
    log_level: Optional[str] = typer.Option(None, help="Override log level (DEBUG/INFO/WARN/ERROR)."),
):
    """
    Paper mode: run the live event loop but do NOT send orders.
    (We’ll implement the runner in src/hyperstat/live/runner.py later.)
    """
    cfg = _common_bootstrap(config, log_level=log_level)
    _print_effective_config(cfg)

    # Basic runtime sanity checks (env vars)
    addr_env = cfg["exchange"]["account"]["address_env"]
    pk_env = cfg["exchange"]["account"]["private_key_env"]
    if not os.getenv(addr_env):
        rprint(f"[yellow]Missing env var {addr_env}. Put it in .env[/yellow]")
    if not os.getenv(pk_env):
        rprint(f"[yellow]Missing env var {pk_env}. Put it in .env[/yellow]")

    # Wire up common objects used by live/paper
    clock = LiveClock()
    ks_cfg = _risk_cfg_from_config(cfg)
    risk_state = RiskState(config=ks_cfg)

    rprint("[green]Paper mode bootstrap OK.[/green]")
    rprint("Next step: implement live/runner.py and exchange/hyperliquid clients.")
    _ = (clock, risk_state)  # keep linters happy


@app.command("live")
def live(
    config: List[str] = typer.Option(
        ["configs/default.yaml", "configs/hyperliquid_mainnet.yaml", "configs/strategy_stat_arb.yaml"],
        help="Config files: default, env override, strategy override (deep-merged).",
    ),
    log_level: Optional[str] = typer.Option(None, help="Override log level (DEBUG/INFO/WARN/ERROR)."),
):
    """
    Live mode: same as paper but execution enabled.
    (We’ll enforce additional safety checks when we implement execution.)
    """
    cfg = _common_bootstrap(config, log_level=log_level)
    _print_effective_config(cfg)

    rprint("[bold red]LIVE MODE NOT IMPLEMENTED YET.[/bold red]")
    rprint("We’ll implement exchange execution and a hard safety gate next.")
    # We still bootstrap risk tracking
    ks_cfg = _risk_cfg_from_config(cfg)
    _ = RiskState(config=ks_cfg)


@app.command("backtest")
def backtest(
    config: List[str] = typer.Option(
        ["configs/default.yaml", "configs/hyperliquid_testnet.yaml", "configs/strategy_stat_arb.yaml"],
        help="Config files: default, env override, strategy override (deep-merged).",
    ),
    log_level: Optional[str] = typer.Option(None, help="Override log level (DEBUG/INFO/WARN/ERROR)."),
):
    """
    Backtest mode: will run historical candles + funding through backtest engine.
    (We’ll implement backtest/engine.py later.)
    """
    cfg = _common_bootstrap(config, log_level=log_level)
    _print_effective_config(cfg)

    rprint("[green]Backtest bootstrap OK.[/green]")
    rprint("Next step: implement data loaders + backtest engine.")
    # Risk state is also used in backtests (cooldown, kill-switch)
    ks_cfg = _risk_cfg_from_config(cfg)
    _ = RiskState(config=ks_cfg)


if __name__ == "__main__":
    app()
