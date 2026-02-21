# src/hyperstat/backtest/reports.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .metrics import PerformanceMetrics, metrics_to_dict


@dataclass(frozen=True)
class BacktestReport:
    equity_curve: pd.DataFrame        # index ts, col equity
    pnl_curve: pd.DataFrame           # index ts, cols: pnl_price, pnl_funding, costs, pnl_net_step
    turnover: pd.Series               # index ts, sum(abs(delta_w))
    weights: pd.DataFrame             # index ts, columns symbols
    metrics: PerformanceMetrics
    breakdown: Dict[str, float]
    meta: Dict[str, str]


# ── Sections for organized display ───────────────────────────────────────────

_METRIC_SECTIONS: Dict[str, list[str]] = {
    "Rendement": [
        "total_return", "cagr",
    ],
    "Ratios performance / risque": [
        "sharpe", "sortino", "calmar", "treynor",
    ],
    "Risque": [
        "ann_vol", "max_drawdown", "avg_drawdown", "max_dd_duration_bars", "var_95",
    ],
    "Robustesse (barre par barre)": [
        "win_rate", "loss_rate", "profit_factor", "avg_gain_loss_ratio", "kelly_fraction",
    ],
    "Exposition": [
        "avg_gross", "avg_net", "avg_turnover",
    ],
    "Corrélation marché": [
        "beta", "r_squared", "jensen_alpha",
    ],
    "PnL décomposé": [
        "pnl_gross", "pnl_funding", "pnl_fees", "pnl_slippage", "pnl_net",
    ],
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_report_csv(report: BacktestReport, out_dir: str) -> None:
    p = Path(out_dir)
    _ensure_dir(p)

    report.equity_curve.to_csv(p / "equity_curve.csv")
    report.pnl_curve.to_csv(p / "pnl_curve.csv")
    report.turnover.to_frame("turnover").to_csv(p / "turnover.csv")
    report.weights.to_csv(p / "weights.csv")

    # métriques formatées
    formatted = metrics_to_dict(report.metrics)
    pd.DataFrame.from_dict(formatted, orient="index", columns=["value"]).to_csv(
        p / "metrics.csv"
    )

    b = pd.DataFrame([report.breakdown]).T
    b.columns = ["value"]
    b.to_csv(p / "breakdown.csv")


def _metrics_sections_html(m: PerformanceMetrics) -> str:
    """Génère les tables HTML organisées par section."""
    formatted = metrics_to_dict(m)
    html_parts = []
    for section, keys in _METRIC_SECTIONS.items():
        rows = ""
        for k in keys:
            if k not in formatted:
                continue
            label = k.replace("_", " ").title()
            rows += f"<tr><td>{label}</td><td><b>{formatted[k]}</b></td></tr>"
        if rows:
            html_parts.append(
                f"<div class='card'>"
                f"<h3>{section}</h3>"
                f"<table><tbody>{rows}</tbody></table>"
                f"</div>"
            )
    return "\n".join(html_parts)


def save_report_html(report: BacktestReport, out_dir: str, title: str = "hyperstat backtest") -> None:
    p = Path(out_dir)
    _ensure_dir(p)

    m = report.metrics
    breakdown_df = pd.DataFrame([report.breakdown]).T
    breakdown_df.columns = ["value"]
    meta_df = pd.DataFrame([report.meta]).T
    meta_df.columns = ["value"]

    metrics_html = _metrics_sections_html(m)

    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>{title}</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8f8f8; }}
          h1 {{ margin-bottom: 4px; }}
          h2 {{ margin-top: 24px; margin-bottom: 8px; }}
          h3 {{ margin: 0 0 8px 0; color: #333; font-size: 14px; text-transform: uppercase;
                letter-spacing: 0.5px; }}
          .subtitle {{ color: #666; font-size: 13px; margin-bottom: 20px; }}
          .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
          .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
          .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 14px;
                   background: #fff; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #eee; padding: 6px 8px; text-align: left;
                    font-size: 13px; }}
          th {{ background: #fafafa; font-weight: 600; }}
          td:last-child {{ text-align: right; font-family: monospace; }}
          .badge-good {{ color: #178a2a; font-weight: bold; }}
          .badge-bad  {{ color: #c0392b; font-weight: bold; }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        <div class="subtitle">
          {m.start.date()} → {m.end.date()} &nbsp;|&nbsp;
          {m.n_steps:,} barres
        </div>

        <h2>Métriques de performance</h2>
        <div class="grid">
          {metrics_html}
        </div>

        <h2>Breakdown PnL</h2>
        <div class="card">
          {breakdown_df.to_html(header=False)}
        </div>

        <h2>Meta</h2>
        <div class="card">
          {meta_df.to_html(header=False)}
        </div>

        <h2>Equity curve (premières barres)</h2>
        <div class="card">
          {report.equity_curve.head(20).to_html()}
        </div>

        <h2>Equity curve (dernières barres)</h2>
        <div class="card">
          {report.equity_curve.tail(20).to_html()}
        </div>
      </body>
    </html>
    """
    (p / "report.html").write_text(html, encoding="utf-8")
