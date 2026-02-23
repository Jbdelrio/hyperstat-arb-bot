"""
hyperstat.agents.utils.news_fetcher
=====================================
Récupère des news crypto depuis des sources gratuites :
    1. Flux RSS  : Coindesk, Cointelegraph (gratuit, sans clé)
    2. CryptoCompare News API (clé gratuite requise, 100k req/mois)

Produit un sentiment score ∈ [-1, 1] basé sur un lexique de mots-clés.
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# ── Lexique de sentiment ──────────────────────────────────────────────────────
_BULLISH_WORDS = {
    "surge", "rally", "breakout", "bullish", "buy", "pump", "moon",
    "adoption", "partnership", "upgrade", "institutional", "etf",
    "approval", "launch", "integration", "milestone", "record", "high",
    "gains", "recovery", "growth", "positive", "optimistic",
}

_BEARISH_WORDS = {
    "crash", "dump", "bearish", "sell", "fear", "hack", "exploit",
    "ban", "regulation", "crackdown", "delist", "warning", "concern",
    "loss", "decline", "drop", "plunge", "liquidation", "scandal",
    "fraud", "lawsuit", "security", "vulnerability", "risk",
}

_RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]

_CC_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"


# ─────────────────────────────────────────────────────────────────────────────

class NewsFetcher:
    """
    Agrège les news depuis RSS et CryptoCompare.
    Calcule un score de sentiment ∈ [-1, 1].

    Usage
    -----
    >>> fetcher = NewsFetcher(cc_api_key=os.getenv("CRYPTOCOMPARE_API_KEY"))
    >>> score   = fetcher.get_sentiment_score(lookback_hours=4)
    >>> news    = fetcher.get_recent_news(lookback_hours=4)
    """

    def __init__(
        self,
        cc_api_key   : Optional[str]  = None,
        timeout      : int            = 8,
        cache_ttl    : int            = 900,   # 15 min
    ):
        self.cc_api_key  = cc_api_key or os.getenv("CRYPTOCOMPARE_API_KEY", "")
        self.timeout     = timeout
        self.cache_ttl   = cache_ttl
        self._cache_news : List[dict] = []
        self._cache_ts   : float      = 0.0

    # ── API publique ──────────────────────────────────────────────────────

    def get_recent_news(self, lookback_hours: int = 4) -> List[dict]:
        """
        Retourne les articles des dernières N heures.

        Chaque article : {title, body, source, ts, sentiment_score}
        """
        self._refresh_cache_if_needed()
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        return [a for a in self._cache_news if a["ts"] >= cutoff]

    def get_sentiment_score(self, lookback_hours: int = 4) -> float:
        """
        Score de sentiment moyen sur les N dernières heures.
        ∈ [-1, 1] : -1 = très bearish, +1 = très bullish.
        """
        articles = self.get_recent_news(lookback_hours)
        if not articles:
            return 0.0
        scores = [a["sentiment_score"] for a in articles]
        # Moyenne pondérée par la confiance (abs du score)
        weights = [abs(s) + 0.01 for s in scores]
        total_w = sum(weights)
        weighted = sum(s * w for s, w in zip(scores, weights))
        return round(weighted / total_w, 4)

    def get_keyword_hits(self, lookback_hours: int = 4) -> dict:
        """Debug : retourne le décompte de mots bullish/bearish détectés."""
        articles = self.get_recent_news(lookback_hours)
        bull_count = bear_count = 0
        for a in articles:
            text = (a["title"] + " " + a.get("body", "")).lower()
            bull_count += sum(1 for w in _BULLISH_WORDS if w in text)
            bear_count += sum(1 for w in _BEARISH_WORDS if w in text)
        return {"bullish_hits": bull_count, "bearish_hits": bear_count, "articles": len(articles)}

    # ── Refresh cache ─────────────────────────────────────────────────────

    def _refresh_cache_if_needed(self):
        now = time.monotonic()
        if (now - self._cache_ts) < self.cache_ttl:
            return
        articles = []
        articles.extend(self._fetch_rss())
        if self.cc_api_key:
            articles.extend(self._fetch_cryptocompare())
        # Déduplique par titre
        seen = set()
        unique = []
        for a in articles:
            key = a["title"][:60]
            if key not in seen:
                seen.add(key)
                a["sentiment_score"] = self._score_article(a)
                unique.append(a)
        self._cache_news = sorted(unique, key=lambda x: x["ts"], reverse=True)
        self._cache_ts   = now

    def _fetch_rss(self) -> List[dict]:
        """Fetch RSS feeds sans librairie externe (parsing XML minimal)."""
        articles = []
        for url in _RSS_FEEDS:
            try:
                resp = requests.get(url, timeout=self.timeout,
                                    headers={"User-Agent": "HyperStat/2.0"})
                resp.raise_for_status()
                # Parsing XML minimal sans feedparser
                items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)
                for item in items[:20]:
                    title = self._extract_tag(item, "title")
                    desc  = self._extract_tag(item, "description")
                    pub   = self._extract_tag(item, "pubDate")
                    ts    = self._parse_rss_date(pub)
                    if title:
                        articles.append({
                            "title" : title,
                            "body"  : desc,
                            "source": url.split("/")[2],
                            "ts"    : ts,
                        })
            except Exception as exc:
                logger.debug(f"[NewsFetcher] RSS {url} échec: {exc}")
        return articles

    def _fetch_cryptocompare(self) -> List[dict]:
        """Fetch CryptoCompare News API (clé gratuite requise)."""
        articles = []
        try:
            resp = requests.get(
                _CC_NEWS_URL,
                params={"lang": "EN", "sortOrder": "latest"},
                headers={"authorization": f"Apikey {self.cc_api_key}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            for item in resp.json().get("Data", [])[:30]:
                articles.append({
                    "title" : item.get("title", ""),
                    "body"  : item.get("body", ""),
                    "source": item.get("source", "cryptocompare"),
                    "ts"    : datetime.utcfromtimestamp(item.get("published_on", 0)),
                })
        except Exception as exc:
            logger.debug(f"[NewsFetcher] CryptoCompare échec: {exc}")
        return articles

    # ── Scoring ───────────────────────────────────────────────────────────

    @staticmethod
    def _score_article(article: dict) -> float:
        """Score ∈ [-1, 1] par comptage de mots-clés."""
        text = (article.get("title", "") + " " + article.get("body", "")).lower()
        bull = sum(1 for w in _BULLISH_WORDS if w in text)
        bear = sum(1 for w in _BEARISH_WORDS if w in text)
        total = bull + bear
        if total == 0:
            return 0.0
        return round((bull - bear) / total, 4)

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        if m:
            val = re.sub(r"<[^>]+>|<!\[CDATA\[|\]\]>", "", m.group(1)).strip()
            return val
        return ""

    @staticmethod
    def _parse_rss_date(date_str: str) -> datetime:
        """Parse RFC 2822 dates (format RSS standard)."""
        if not date_str:
            return datetime.utcnow()
        fmts = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
        ]
        for fmt in fmts:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.replace(tzinfo=None)
            except ValueError:
                continue
        return datetime.utcnow()
