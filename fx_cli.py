#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FX Signal & Forecast CLI (USD/JPY/CNY vs KRW)

- 데이터 소스:
  - API: exchangerate.host (기본) + frankfurter.app (폴백)
  - HTML: 사용자가 구현하는 크롤링 소스 (예: 환율 페이지)
- 지표: SMA(20/60), RSI(14)
- 신호: SMA 교차 + RSI 필터
- 예측: ARIMA(1,1,1)
- 출력: 콘솔 요약 + CSV + PNG
"""

import argparse
import os
from datetime import date
from typing import Dict, List

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# -------------------- 설정 --------------------
START_DATE = "2018-01-01"
BASE = "USD"
SYMBOLS = ["KRW", "JPY", "CNY"]  # API용
FAST = 20
SLOW = 60
RSI_N = 14

PAIRS_READABLE = {
    "USDKRW": "미국 달러 (USD/KRW)",
    "JPYKRW": "일본 엔 (JPY/KRW)",
    "CNYKRW": "중국 위안 (CNY/KRW)",
}

# -------------------- 공통 후처리 --------------------
def _postprocess_usd_base_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    USD 기준 KRW/JPY/CNY 시계열을 받아서
    - 인덱스를 일단위로 맞추고
    - 원화 기준 엔/위안 교차환율 컬럼을 만든다.
    기대 컬럼: KRW, JPY, CNY (각각 1 USD 당 통화 금액)
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().asfreq("D").ffill()

    # USD base 컬럼 통일
    df.rename(columns={"KRW": "USDKRW", "JPY": "USDJPY", "CNY": "USDCNY"}, inplace=True)

    # 원화 기준 엔/위안 (1 엔 / 1 위안당 원화)
    df["JPYKRW"] = df["USDKRW"] / df["USDJPY"]
    df["CNYKRW"] = df["USDKRW"] / df["USDCNY"]

    return df[["USDKRW", "JPYKRW", "CNYKRW"]]


# -------------------- 데이터 소스 인터페이스 --------------------
class FXDataSource:
    """USDKRW, JPYKRW, CNYKRW 시계열을 반환하는 공통 인터페이스"""

    def get_timeseries(self, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError


# -------------------- API 기반 데이터 소스 --------------------
class ApiDataSource(FXDataSource):
    """exchangerate.host + frankfurter.app 기반"""

    def __init__(self, base: str = BASE, symbols: List[str] = None):
        if symbols is None:
            symbols = SYMBOLS
        self.base = base
        self.symbols = symbols

    def _fetch_from_exchangerate_host(self, start: str, end: str) -> pd.DataFrame:
        url = "https://api.exchangerate.host/timeseries"
        params = {"start_date": start, "end_date": end, "base": self.base, "symbols": ",".join(self.symbols)}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        rates = j.get("rates", {})
        if not rates:
            raise RuntimeError("exchangerate.host 응답에 rates가 비어 있음")
        df = pd.DataFrame.from_dict(rates, orient="index")
        print("✔ 데이터 소스: exchangerate.host")
        return _postprocess_usd_base_df(df)

    def _fetch_from_frankfurter(self, start: str, end: str) -> pd.DataFrame:
        url = f"https://api.frankfurter.app/{start}..{end}"
        params = {"from": "USD", "to": "KRW,JPY,CNY"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        rates = j.get("rates", {})
        if not rates:
            raise RuntimeError("frankfurter.app 응답에 rates가 비어 있음")
        df = pd.DataFrame.from_dict(rates, orient="index")
        print("✔ 데이터 소스: frankfurter.app")
        return _postprocess_usd_base_df(df)

    def get_timeseries(self, start_date: str, end_date: str) -> pd.DataFrame:
        # 1순위: exchangerate.host
        try:
            return self._fetch_from_exchangerate_host(start_date, end_date)
        except Exception as e:
            print(f"ℹ exchangerate.host 실패: {e} → frankfurter.app로 폴백")
        # 2순위: frankfurter.app
        return self._fetch_from_frankfurter(start_date, end_date)


# -------------------- HTML 기반 데이터 소스 (사용자 구현 자리) --------------------
class HtmlDataSource(FXDataSource):
    """
    HTML 환율 페이지(예: 어떤 금융 포털의 환율 화면 등)를 직접 파싱해서
    USDKRW, JPYKRW, CNYKRW 값을 가져오는 소스.

    과제/연구용으로, 사용자가 직접 크롤링 코드를 채워 넣는 구조로 설계하였다.
    이 클래스 내부에 requests + HTML 파서(BeautifulSoup 등)를 이용해
    현재 시점의 USD/KRW, JPY/KRW, CNY/KRW 값을 추출하는 로직을 구현하면 된다.
    """

    def __init__(self):
        pass

    def get_latest_quote(self) -> Dict[str, float]:
        """
        현재 시점의 1 USD당 KRW/JPY/CNY 시세를 딕셔너리로 반환하는 자리.

        예시 형식:
        {
            "USDKRW": 1385.25,
            "USDJPY": 151.32,
            "USDCNY": 7.12
        }

        이후 _postprocess_usd_base_df()에서 JPYKRW, CNYKRW로 변환된다.

        여기 안에 HTML 요청 및 파싱 코드를 직접 작성하면 된다.
        """
        # TODO: 사용자가 직접 구현
        raise NotImplementedError("HTML 크롤링 기반 최신 환율 가져오기 로직을 여기에 구현하세요.")

    def get_timeseries(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        순수 HTML만으로 과거 시계를 쌓는 구조는 현실성이 떨어지므로,
        과제 관점에서는 다음 두 가지 패턴 중 하나로 설계할 수 있다.

        1) '오늘' 1일치만 반환하는 단기 데모
        2) 외부 CSV/기존 API 시계열 + 오늘 하루를 HTML로 보정

        여기서는 1)번, 오늘 하루만 담은 시계열 예시 설계만 제공한다.
        """
        latest = self.get_latest_quote()  # {"USDKRW": ..., "USDJPY": ..., "USDCNY": ...}
        df = pd.DataFrame(
            {
                "KRW": [latest["USDKRW"]],
                "JPY": [latest["USDJPY"]],
                "CNY": [latest["USDCNY"]],
            },
            index=[pd.to_datetime(end_date)],
        )
        return _postprocess_usd_base_df(df)


# -------------------- 지표/신호/예측 --------------------
def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def compute_indicators(df: pd.DataFrame, pair: str,
                       fast: int = FAST, slow: int = SLOW, rsi_n: int = RSI_N) -> pd.DataFrame:
    s = df[pair].copy()
    out = pd.DataFrame(index=df.index)
    out["price"] = s
    out["sma_fast"] = s.rolling(fast, min_periods=1).mean()
    out["sma_slow"] = s.rolling(slow, min_periods=1).mean()
    out["rsi"] = rsi(s, rsi_n)

    cross_up = (out["sma_fast"] > out["sma_slow"]) & (out["sma_fast"].shift(1) <= out["sma_slow"].shift(1))
    cross_down = (out["sma_fast"] < out["sma_slow"]) & (out["sma_fast"].shift(1) >= out["sma_slow"].shift(1))

    sig = pd.Series("HOLD", index=out.index, dtype=object)
    sig = np.where(cross_up & (out["rsi"] < 70), "BUY", sig)
    sig = np.where(cross_down & (out["rsi"] > 30), "SELL", sig)
    out["signal"] = pd.Categorical(sig, categories=["SELL", "HOLD", "BUY"])
    return out


def backtest_signals(tbl: pd.DataFrame):
    """
    - 최근 비-HOLD가 BUY면 보유(1), SELL이면 비보유(0)
    - 보유일에만 일간 수익률(pct_change) 반영
    """
    last = None
    pos = []
    for sig in tbl["signal"]:
        if sig in ("BUY", "SELL"):
            last = sig
        pos.append(1 if last == "BUY" else 0)
    pos = pd.Series(pos, index=tbl.index)
    ret = tbl["price"].pct_change().fillna(0) * pos
    equity = (1 + ret).cumprod()
    return pos, ret, equity


def arima_forecast(series: pd.Series, steps: int = 7):
    """ARIMA(1,1,1) 기반 로그 스케일 예측"""
    y = np.log(series.dropna())
    model = ARIMA(y, order=(1, 1, 1))
    fit = model.fit()
    fc_log = fit.forecast(steps=steps)
    fc = np.exp(fc_log)
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return pd.Series(fc, index=idx)


def recommendation(sig: str) -> str:
    if sig == "BUY":
        return "외화 매수(원화 약세 대비)"
    if sig == "SELL":
        return "외화 매도/환전(원화 강세 대비)"
    return "관망(HOLD)"


def save_chart(tbl: pd.DataFrame, pair: str, outfile: str):
    plt.figure(figsize=(10, 4))
    plt.plot(tbl.index, tbl["price"], label="Price")
    plt.plot(tbl.index, tbl["sma_fast"], label=f"SMA{FAST}")
    plt.plot(tbl.index, tbl["sma_slow"], label=f"SMA{SLOW}")
    latest_sig = str(tbl["signal"].iloc[-1])
    plt.title(f"{pair} (latest signal: {latest_sig})")
    plt.xlabel("Date")
    plt.ylabel("KRW per unit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


# -------------------- 메인 --------------------
def main():
    parser = argparse.ArgumentParser(description="FX Signal & Forecast CLI (USD/JPY/CNY vs KRW)")
    parser.add_argument("--pairs", type=str, default="USDKRW,JPYKRW,CNYKRW",
                        help="분석할 페어(콤마로 구분): USDKRW,JPYKRW,CNYKRW")
    parser.add_argument("--horizon", type=int, default=7, help="예측일수 (기본 7일)")
    parser.add_argument("--outdir", type=str, default="./out", help="결과 저장 폴더")
    parser.add_argument("--source", type=str, default="api",
                        choices=["api", "html", "api+html"],
                        help="데이터 소스 선택: api / html / api+html")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    end = date.today().isoformat()

    api_source = ApiDataSource()
    html_source = HtmlDataSource()

    # 데이터 소스 선택
    if args.source == "api":
        df = api_source.get_timeseries(START_DATE, end)
    elif args.source == "html":
        # HtmlDataSource.get_latest_quote()를 직접 구현해야 동작한다.
        df = html_source.get_timeseries(START_DATE, end)
    else:  # api+html
        df = api_source.get_timeseries(START_DATE, end)
        try:
            latest = html_source.get_latest_quote()
            # 오늘 날짜 인덱스 맞추기
            today = pd.to_datetime(end)
            if today not in df.index:
                df.loc[today] = df.iloc[-1]
                df = df.sort_index()
            # HTML 기준으로 오늘 값 덮어쓰기
            tmp = pd.DataFrame(
                {
                    "KRW": [latest["USDKRW"]],
                    "JPY": [latest["USDJPY"]],
                    "CNY": [latest["USDCNY"]],
                },
                index=[today],
            )
            df.loc[today] = _postprocess_usd_base_df(tmp).iloc[0]
            print("ℹ 오늘 날짜 환율을 HTML 데이터로 보정함")
        except NotImplementedError:
            print("ℹ HtmlDataSource.get_latest_quote()가 구현되지 않아 API 데이터만 사용합니다.")
        except Exception as e:
            print(f"ℹ HTML 보정 실패: {e} → API 데이터만 사용")

    # 페어 루프
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    print(f"\n=== 수집기간: {df.index[0].date()} ~ {df.index[-1].date()} | 페어: {pairs} ===")

    for pair in pairs:
        if pair not in ["USDKRW", "JPYKRW", "CNYKRW"]:
            print(f"[건너뜀] 지원하지 않는 페어: {pair}")
            continue

        # 지표/신호
        t = compute_indicators(df, pair, FAST, SLOW, RSI_N)
        pos, ret, eq = backtest_signals(t)
        t["position"] = pos
        t["equity"] = eq

        # 예측
        fc = arima_forecast(t["price"], steps=args.horizon)

        # 요약 출력
        last_px = float(t["price"].iloc[-1])
        last_sig = str(t["signal"].iloc[-1])
        reco = recommendation(last_sig)
        if len(t) > 252:
            perf_1y = float(t["equity"].iloc[-1] / t["equity"].iloc[-252] - 1)
            perf_1y_str = f"{perf_1y * 100:.2f}%"
        else:
            perf_1y_str = "N/A"

        readable = PAIRS_READABLE.get(pair, pair)
        print(f"\n[{pair}] {readable}")
        print(f"- 현재 환율: {last_px:.2f} KRW")

        # 출력 문구를 더 자연스럽게
        if last_sig == "BUY":
            action_text = "지금은 외화를 사는 게 좋습니다"
        elif last_sig == "SELL":
            action_text = "지금은 외화를 파는 게 좋습니다"
        else:
            action_text = "지금은 기다리는 게 좋습니다"

        print(f"- 현재 추천 : {last_sig} ({action_text})")
        print(f"- 최근 1년 모의 수익률 : {perf_1y_str}")

        print(f"- {args.horizon}일 예측(첫 3일):")
        for d, v in list(fc.items())[:3]:
            print(f"  · {d.date()}: {v:.2f}")

        # 저장물
        sig_path = os.path.join(args.outdir, f"{pair}_signals.csv")
        fc_path = os.path.join(args.outdir, f"{pair}_forecast_{args.horizon}d.csv")
        png_path = os.path.join(args.outdir, f"{pair}_chart.png")
        t.to_csv(sig_path, index=True, encoding="utf-8")
        fc.to_csv(fc_path, header=[f"{pair}_forecast"], encoding="utf-8")
        save_chart(t, pair, png_path)

    print(f"\n✅ 저장 폴더: {os.path.abspath(args.outdir)}")
    print(" - *_signals.csv     : 가격/지표/신호/포지션/누적지수")
    print(" - *_forecast_*d.csv : ARIMA 예측 값")
    print(" - *_chart.png       : 가격+SMA 차트")


if __name__ == "__main__":
    main()
