import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import urllib.parse
import xml.etree.ElementTree as ET
import requests
import re

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

st.set_page_config(page_title="AI 주식 분석 및 예측 시스템", layout="wide")

# =========================
# 기본 설정
# =========================
torch.manual_seed(42)
np.random.seed(42)


# =========================
# 웹 스타일
# =========================
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    div[data-baseweb="input"] { background-color: #F3F4F6 !important; border-radius: 8px; border: 2px solid #4F46E5 !important; }
    input { color: #FFFFFF !important; font-weight: bold !important; font-size: 16px !important; background-color: #111111 !important; }
    div[data-baseweb="input"] input::placeholder { color: #AAAAAA !important; }
    h1, h2, h3, h4, h5 { font-family: 'Noto Sans KR', sans-serif; font-weight: 700; color: #FFFFFF; }
    .search-container { padding: 40px 0; background: linear-gradient(180deg, #0A0A0A 0%, #000000 100%); border-bottom: 1px solid #222; margin-bottom: 20px; }
    .sub-text { color: #FFFFFF !important; font-size: 14px; text-align: center; margin-top: 5px; }
    div[data-testid="stRadio"] label, div[data-testid="stRadio"] p, div[data-testid="stRadio"] span { color: #FFFFFF !important; font-weight: 600 !important; font-size: 16px !important; }
    
    .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 20px; }
    .info-card { background-color: #111111; border: 1px solid #333333; padding: 15px; border-radius: 8px; text-align: center; }
    .info-label { color: #AAAAAA !important; font-size: 12px; margin-bottom: 6px; font-weight: 500; }
    .info-value { color: #FFFFFF !important; font-size: 18px; font-weight: 700; }
    
    .metric-card { background-color: #111111; border: 1px solid #222222; padding: 20px; border-radius: 12px; text-align: center; }
    .metric-title { color: #FFFFFF !important; font-size: 16px; font-weight: 600; margin-bottom: 10px; }
    .metric-value { color: #FFFFFF !important; font-size: 32px; font-weight: 700; margin-bottom: 5px; }
    .metric-delta { color: #FF3232 !important; font-size: 16px; font-weight: 600; }
    .metric-delta.down { color: #0062FF !important; }
    
    .stTable, table, th, td, tr { color: #FFFFFF !important; font-weight: 500 !important; border-color: #333333 !important; }
    table th { background-color: #1F1F1F !important; font-weight: 700 !important; color: #00FFCC !important; }
    table td { background-color: #0E0E0E !important; }
    
    .news-card { background-color: #111111; border: 1px solid #222222; border-radius: 10px; padding: 20px; margin-bottom: 16px; transition: 0.2s; }
    .news-card:hover { border-color: #4F46E5; background-color: #151515; }
    .news-title { font-size: 18px; font-weight: 700; color: #00FFCC !important; text-decoration: none; margin-bottom: 8px; display: block; }
    .news-meta { font-size: 12px; color: #888888; margin-bottom: 12px; border-bottom: 1px solid #222; padding-bottom: 8px; }
    .news-summary-box { font-size: 14.5px; color: #DDDDDD; line-height: 1.7; padding-left: 12px; border-left: 3px solid #4F46E5; text-align: justify; word-break: break-all; }
    </style>
    """, unsafe_allow_html=True)


# =========================
# 데이터 수집 함수
# =========================
@st.cache_data(ttl=600)
def fetch_stock_data(ticker, period_code, interval_code):
    for attempt in range(3):
        try:
            data = yf.download(ticker, period=period_code, interval=interval_code, progress=False, auto_adjust=False)
            if not data.empty:
                return data
        except Exception:
            time.sleep(1.0)
    return yf.download(ticker, period=period_code, interval=interval_code, progress=False, auto_adjust=False)


@st.cache_data(ttl=1800)
def fetch_ticker_info(ticker):
    t = yf.Ticker(ticker)
    try:
        f_info = dict(t.fast_info)
    except Exception:
        f_info = {}
    try:
        i_info = dict(t.info)
    except Exception:
        i_info = {}
    return f_info, i_info


@st.cache_data(ttl=300)
def get_live_indices():
    indices_data = []
    tickers = {
        "📊 KOSPI 지수": "^KS11",
        "📈 NASDAQ 100": "^NDX",
        "🪙 BITCOIN (BTC)": "BTC-USD"
    }

    for name, ticker in tickers.items():
        try:
            t_data = yf.download(ticker, period="2d", interval="1d", progress=False, auto_adjust=False)
            if len(t_data) >= 2:
                closes = t_data["Close"].to_numpy().flatten()
                current = float(closes[-1])
                prev = float(closes[-2])
                change_pct = ((current - prev) / prev) * 100
                is_down = change_pct < 0
                arrow = "▼" if is_down else "▲"
                sign = "" if is_down else "+"
                indices_data.append((name, f"{current:,.2f}", f"{sign}{change_pct:.2f}% {arrow}", is_down))
            else:
                indices_data.append((name, "로딩중", "-", False))
        except Exception:
            indices_data.append((name, "N/A", "-", False))

    return indices_data


# =========================
# 종목 검색
# =========================
def search_stock_suggestions(query):
    clean_query = query.strip()
    if not clean_query:
        return []

    korean_market_dict = {
        "삼성전자": "005930.KS", "삼성": "005930.KS", "삼성전자우": "005935.KS",
        "하이닉스": "000660.KS", "sk하이닉스": "000660.KS", "에스케이하이닉스": "000660.KS",
        "현대차": "005380.KS", "현대자동차": "005380.KS", "기아": "000270.KS",
        "네이버": "035420.KS", "naver": "035420.KS", "카카오": "035720.KS",
        "포스코": "005490.KS", "포스코홀딩스": "005490.KS", "posco홀딩스": "005490.KS",
        "에코프로": "086520.KQ", "에코프로비엠": "247540.KQ", "엘지에너지솔루션": "373220.KS",
        "lg에너지솔루션": "373220.KS", "셀트리온": "068270.KS", "알테오জেন": "196170.KQ",
        "애플": "AAPL", "apple": "AAPL", "테슬라": "TSLA", "tesla": "TSLA",
        "엔비디아": "NVDA", "nvidia": "NVDA", "마이크로소프트": "MSFT", "구글": "GOOGL",
        "아마존": "AMZN", "비트코인": "BTC-USD", "이더리움": "ETH-USD"
    }

    suggestions = []
    lower_query = clean_query.lower()

    for kor_key, ticker_val in korean_market_dict.items():
        if kor_key in lower_query or lower_query in kor_key:
            suggestions.append(f"{kor_key} ({ticker_val})")

    suggestions = list(dict.fromkeys(suggestions))

    try:
        search_obj = yf.Search(clean_query, max_results=5)
        search_res = getattr(search_obj, "tickers", [])

        if not search_res and hasattr(search_obj, "quotes"):
            search_res = search_obj.quotes

        if search_res:
            for item in search_res:
                if not isinstance(item, dict):
                    continue

                symbol = item.get("symbol", "")
                name = item.get("shortname", item.get("longname", item.get("name", "주식 자산")))
                q_type = item.get("quoteType", "")

                if symbol and q_type in ["EQUITY", "ETF", "CRYPTOCURRENCY"]:
                    option_str = f"{name} ({symbol})"
                    if option_str not in suggestions:
                        suggestions.append(option_str)
    except Exception:
        pass

    if not suggestions:
        suggestions.append(clean_query.upper())

    return suggestions


# =========================
# 뉴스 수집
# =========================
@st.cache_data(ttl=900)
def get_korean_real_news(keyword):
    encoded_keyword = urllib.parse.quote(keyword)
    url = f"https://news.google.com/rss/search?q={encoded_keyword}&hl=ko&gl=KR&ceid=KR:ko"
    news_items = []

    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.text)

        for item in root.findall(".//item"):
            title_node = item.find("title")
            link_node = item.find("link")
            date_node = item.find("pubDate")
            desc_node = item.find("description")

            if title_node is None or link_node is None:
                continue

            title = title_node.text or "제목 없음"
            link = link_node.text or "#"
            pub_date = date_node.text if date_node is not None else "날짜 정보 없음"
            raw_desc = desc_node.text if desc_node is not None else ""
            clean_desc = re.sub(r"<[^>]+>", "", raw_desc).strip()

            source = "금융 경제 뉴스"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                source = parts[1]

            if len(clean_desc) < 20:
                clean_desc = "뉴스 요약 정보가 충분하지 않습니다. 제목을 클릭해 원문을 확인하세요."

            is_duplicate = False
            for existing in news_items:
                if len(set(title.split()) & set(existing["title"].split())) > 3:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            news_items.append({
                "title": title,
                "link": link,
                "date": pub_date,
                "source": source,
                "desc": clean_desc
            })

            if len(news_items) >= 3:
                break
    except Exception:
        pass

    return news_items[:3]


# =========================
# 🆕 뉴스 감성 스코어 산출 엔진 추가
# =========================
def analyze_news_sentiment(news_items):
    if not news_items:
        return 0.0
    pos_keywords = ['상승', '호실적', '최고', '돌파', '흑자', '성장', '호재', '급등', '추천', '매수', '이익']
    neg_keywords = ['하락', '적자', '감소', '우려', '악재', '부진', '급락', '위기', '쇼크', '하회', '매도']
    
    score = 0.0
    for item in news_items:
        full_text = item['title'] + " " + item['desc']
        for p in pos_keywords:
            if p in full_text: score += 0.25
        for n in neg_keywords:
            if n in full_text: score -= 0.25
    return max(min(score, 1.0), -1.0)


# =========================
# AI 모델 유틸 함수 (기존 로직 유지)
# =========================
def make_sequences(data, window_size=30):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def get_nice_y_range(values, top_padding=0.22, bottom_padding=0.12):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None

    y_min = float(np.min(arr))
    y_max = float(np.max(arr))
    gap = y_max - y_min

    if gap <= 0:
        gap = abs(y_max) * 0.05 if y_max != 0 else 1

    return [y_min - gap * bottom_padding, y_max + gap * top_padding]


def stabilize_prediction(pred, last_price, max_daily_move=0.07):
    pred = np.array(pred, dtype=float)
    if len(pred) == 0:
        return pred

    pred = np.nan_to_num(pred, nan=last_price, posinf=last_price, neginf=last_price)

    offset = last_price - pred[0]
    pred = pred + offset

    fixed = []
    prev = last_price
    for value in pred:
        upper = prev * (1 + max_daily_move)
        lower = prev * (1 - max_daily_move)
        value = min(max(value, lower), upper)
        fixed.append(value)
        prev = value

    return np.array(fixed)


def calculate_rsi(close, period=14):
    close = pd.Series(close).astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def make_momentum_ai_forecast(price_df, base_pred, steps=30):
    data = price_df.copy()
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data["Volume"] = pd.to_numeric(data.get("Volume", 0), errors="coerce")
    data = data.dropna(subset=["Close"])

    close = data["Close"]
    volume = data["Volume"].fillna(0)
    last_price = float(close.iloc[-1])

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    rsi = calculate_rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    recent_return_5 = close.pct_change(5).iloc[-1] if len(close) > 5 else 0
    recent_return_20 = close.pct_change(20).iloc[-1] if len(close) > 20 else 0
    recent_return_60 = close.pct_change(60).iloc[-1] if len(close) > 60 else 0

    volume_avg_5 = volume.rolling(5).mean().iloc[-1] if len(volume) > 5 else volume.iloc[-1]
    volume_avg_20 = volume.rolling(20).mean().iloc[-1] if len(volume) > 20 else volume_avg_5
    volume_ratio = volume_avg_5 / volume_avg_20 if volume_avg_20 and volume_avg_20 > 0 else 1

    score = 50

    if ma5.iloc[-1] > ma20.iloc[-1]: score += 10
    else: score -= 10

    if ma20.iloc[-1] > ma60.iloc[-1]: score += 12
    else: score -= 12

    if recent_return_5 > 0: score += 8
    else: score -= 8

    if recent_return_20 > 0: score += 10
    else: score -= 10

    if recent_return_60 > 0: score += 8
    else: score -= 8

    if macd.iloc[-1] > macd_signal.iloc[-1]: score += 10
    else: score -= 10

    latest_rsi = float(rsi.iloc[-1])
    if 45 <= latest_rsi <= 70: score += 8
    elif latest_rsi > 80: score -= 8
    elif latest_rsi < 35: score += 5

    if volume_ratio > 1.2 and recent_return_5 > 0: score += 10
    elif volume_ratio > 1.2 and recent_return_5 < 0: score -= 8

    score = max(0, min(100, score))

    base_pred = np.array(base_pred, dtype=float)
    base_pred = stabilize_prediction(base_pred, last_price, max_daily_move=0.08)

    momentum_strength = (score - 50) / 50
    daily_bias = momentum_strength * 0.004

    trend_pred = []
    prev = last_price
    for i in range(steps):
        base_target = base_pred[i]
        base_return = (base_target - prev) / prev if prev > 0 else 0

        momentum_decay = 0.96 ** i
        adjusted_return = base_return * 0.55 + daily_bias * momentum_decay

        max_move = 0.09 if score >= 65 else 0.06
        adjusted_return = min(max(adjusted_return, -max_move), max_move)

        next_price = prev * (1 + adjusted_return)
        trend_pred.append(next_price)
        prev = next_price

    return np.array(trend_pred), score, {
        "rsi": latest_rsi,
        "macd": float(macd.iloc[-1]),
        "macd_signal": float(macd_signal.iloc[-1]),
        "volume_ratio": float(volume_ratio),
        "return_5": float(recent_return_5 * 100),
        "return_20": float(recent_return_20 * 100),
        "return_60": float(recent_return_60 * 100),
    }


@st.cache_data(show_spinner=False)
def forecast_arima_cached(close_tuple, steps=30):
    close_prices = np.array(close_tuple, dtype=float)
    series = pd.Series(close_prices)

    try:
        model = ARIMA(series, order=(5, 1, 0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps)
        result = forecast.to_numpy()
    except Exception:
        result = np.repeat(close_prices[-1], steps)

    result = np.nan_to_num(result, nan=close_prices[-1], posinf=close_prices[-1], neginf=close_prices[-1])
    return result


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class PriceTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.output_layer(x)


@st.cache_data(show_spinner=False)
def forecast_lstm_cached(close_tuple, steps=30, window_size=30, epochs=35):
    close_prices = np.array(close_tuple, dtype=float).reshape(-1, 1)
    last_price = float(close_prices[-1][0])

    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_prices)

        X, y = make_sequences(scaled, window_size)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        last_window = torch.tensor(scaled[-window_size:].reshape(1, window_size, 1), dtype=torch.float32)
        preds = []

        with torch.no_grad():
            for _ in range(steps):
                pred = model(last_window).item()
                preds.append(pred)
                new_value = torch.tensor([[[pred]]], dtype=torch.float32)
                last_window = torch.cat([last_window[:, 1:, :], new_value], dim=1)

        preds = np.array(preds).reshape(-1, 1)
        result = scaler.inverse_transform(preds).flatten()
    except Exception:
        result = np.repeat(last_price, steps)

    result = np.nan_to_num(result, nan=last_price, posinf=last_price, neginf=last_price)
    return result


@st.cache_data(show_spinner=False)
def forecast_transformer_cached(close_tuple, steps=30, window_size=30, epochs=45):
    close_prices = np.array(close_tuple, dtype=float).reshape(-1, 1)
    last_price = float(close_prices[-1][0])

    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_prices)

        X, y = make_sequences(scaled, window_size)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        model = PriceTransformer()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        last_window = torch.tensor(scaled[-window_size:].reshape(1, window_size, 1), dtype=torch.float32)
        preds = []

        with torch.no_grad():
            for _ in range(steps):
                pred = model(last_window).item()
                preds.append(pred)
                new_value = torch.tensor([[[pred]]], dtype=torch.float32)
                last_window = torch.cat([last_window[:, 1:, :], new_value], dim=1)

        preds = np.array(preds).reshape(-1, 1)
        result = scaler.inverse_transform(preds).flatten()
    except Exception:
        result = np.repeat(last_price, steps)

    result = np.nan_to_num(result, nan=last_price, posinf=last_price, neginf=last_price)
    return result


# =========================
# 헤더
# =========================
st.markdown("""
    <div class="search-container">
        <h1 style='text-align: center; font-size: 36px; margin-bottom: 10px;'>
            📈 <span style='color: #4F46E5;'>PRO</span> AI CHART ANALYSIS SYSTEM
        </h1>
        <p style='text-align: center; color: #AAAAAA; font-size: 15px;'>ARIMA · LSTM · Transformer · 뉴스 감성 배당 하이브리드 미래 예측</p>
    </div>
    """, unsafe_allow_html=True)


col1, col2, col3 = st.columns([1.2, 2, 1.2])
user_input = None
company_clean_name = "삼성전자"

with col2:
    search_query = st.text_input(
        "주식 검색창",
        placeholder="기업 이름 또는 티커 입력 (예: 삼성전자, Apple, TSLA)",
        label_visibility="collapsed"
    )

    if search_query:
        with st.spinner("연관 주식 리스트 탐색 중..."):
            options = search_stock_suggestions(search_query)

        if options:
            selected_option = st.selectbox("📌 아래 검색 결과에서 분석할 주식을 선택하세요:", options)
            if "(" in selected_option:
                user_input = selected_option.split("(")[-1].replace(")", "").strip()
                company_clean_name = selected_option.split("(")[0].strip()
            else:
                user_input = selected_option.strip()
                company_clean_name = selected_option.strip()
    else:
        st.markdown(
            "<p class='sub-text'>기업 이름이나 티커를 입력하면 연관 주식 선택지가 아래에 나타납니다.</p>",
            unsafe_allow_html=True
        )


# =========================
# 메인 화면
# =========================
if user_input:
    period_map = {"1일": "1d", "1개월": "1mo", "6개월": "6mo", "1년": "1y", "5년": "5y"}
    interval_map = {"1일": "1m", "1개월": "1d", "6개월": "1d", "1년": "1d", "5년": "1d"}

    selected_period_label = st.radio(" 📊 차트 조회 기간 선택", list(period_map.keys()), horizontal=True)
    is_daily_chart = selected_period_label != "1일"

    try:
        with st.spinner("차트 및 뉴스 데이터 동기화 중..."):
            p_code = "5y" if is_daily_chart else period_map[selected_period_label]
            i_code = interval_map[selected_period_label]

            df_raw = fetch_stock_data(user_input, p_code, i_code)
            df_ai = fetch_stock_data(user_input, "2y", "1d")
            fast_info, info_dict = fetch_ticker_info(user_input)
            
            # 실시간 뉴스 크롤링 및 감성 스코어 연동
            news_items = get_korean_real_news(company_clean_name)
            sentiment_score = analyze_news_sentiment(news_items)

        if df_raw.empty or df_ai.empty:
            st.error("⚠️ 데이터를 불러오지 못했습니다. 티커명이 올바른지 확인해 주세요.")
        else:
            df = df_raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            ticker_upper = user_input.upper()
            is_korea = ticker_upper.endswith(".KS") or ticker_upper.endswith(".KQ") or ticker_upper.isdigit()

            if df.index.tzinfo is not None:
                target_tz = "Asia/Seoul" if is_korea else "America/New_York"
                df.index = df.index.tz_convert(target_tz)

            if is_daily_chart:
                df.index = df.index.tz_localize(None)

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    if isinstance(df[col], pd.DataFrame):
                        df[col] = df[col].iloc[:, 0]
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["Open", "High", "Low", "Close"])

            if not is_daily_chart:
                df["Close"] = df["Close"].interpolate(method="linear")

            if is_daily_chart:
                df["MA5"] = df["Close"].rolling(window=5).mean()
                df["MA20"] = df["Close"].rolling(window=20).mean()
                df["MA60"] = df["Close"].rolling(window=60).mean()

                if selected_period_label == "1개월": df = df.iloc[-24:]
                elif selected_period_label == "6개월": df = df.iloc[-130:]
                elif selected_period_label == "1년": df = df.iloc[-252:]

            if len(df) > 0:
                v_open = float(df["Open"].to_numpy().flatten()[0])
                v_high = float(df["High"].to_numpy().flatten().max())
                v_low = float(df["Low"].to_numpy().flatten().min())
                v_close = float(df["Close"].to_numpy().flatten()[-1])
                period_return = ((v_close - v_open) / v_open) * 100
                return_sign = "+" if period_return >= 0 else ""
            else:
                v_open = v_high = v_low = v_close = period_return = 0
                return_sign = ""

            currency = info_dict.get("currency", "USD")
            market_cap = info_dict.get("marketCap", info_dict.get("market_cap", fast_info.get("marketCap", 0)))
            per = info_dict.get("trailingPE", np.nan)

            if market_cap and market_cap > 1e12: mc_str = f"{market_cap / 1e12:,.1f}조 {currency}"
            elif market_cap and market_cap > 1e8: mc_str = f"{market_cap / 1e8:,.0f}억 {currency}"
            elif market_cap and market_cap > 0: mc_str = f"{market_cap:,.0f} {currency}"
            else: mc_str = "N/A"

            per_str = f"{per:,.2f}배" if not pd.isna(per) else "N/A"

            # 상단 정보 그리드 출력
            st.markdown(f"""
                <div class="info-grid">
                    <div class="info-card"><div class="info-label">시가 Open</div><div class="info-value">{v_open:,.2f} {currency}</div></div>
                    <div class="info-card"><div class="info-label">최고가 High</div><div class="info-value" style="color: #FF3232 !important;">{v_high:,.2f} {currency}</div></div>
                    <div class="info-card"><div class="info-label">최저가 Low</div><div class="info-value" style="color: #0062FF !important;">{v_low:,.2f} {currency}</div></div>
                    <div class="info-card"><div class="info-label">시가총액</div><div class="info-value">{mc_str}</div></div>
                    <div class="info-card"><div class="info-label">PER</div><div class="info-value">{per_str}</div></div>
                    <div class="info-card"><div class="info-label">{selected_period_label} 수익률</div><div class="info-value">{return_sign}{period_return:.2f}%</div></div>
                </div>
            """, unsafe_allow_html=True)

            comp_full_name = info_dict.get("longName", info_dict.get("shortName", ""))
            if not comp_full_name:
                try:
                    backup_search = yf.Search(user_input, max_results=1).quotes
                    if backup_search: comp_full_name = backup_search[0].get("longname", backup_search[0].get("shortname", ""))
                except Exception: pass
            if not comp_full_name: comp_full_name = company_clean_name if company_clean_name != ticker_upper else ticker_upper

            if len(df) >= 2:
                close_arr = df["Close"].to_numpy().flatten()
                day_change = float(close_arr[-1] - close_arr[-2])
                day_change_pct = float((day_change / close_arr[-2]) * 100)
            else:
                day_change = day_change_pct = 0.0

            c_sign, c_arrow = ("+", "▲") if day_change >= 0 else ("", "▼")
            c_color = "#FF3232" if day_change >= 0 else "#0062FF"

            st.markdown(f"""
                <div style='margin-bottom: 25px; padding: 15px 20px; background: linear-gradient(90deg, #111111 0%, #000000 100%); border-left: 5px solid #4F46E5; border-radius: 8px;'>
                    <div style='display: flex; align-items: baseline; gap: 10px;'>
                        <span style='font-size: 38px; font-weight: 900; color: #FFFFFF;'>{comp_full_name}</span>
                        <span style='font-size: 16px; font-weight: 700; color: #A0AEC0; background-color: #2D3748; padding: 3px 10px; border-radius: 6px;'>{ticker_upper}</span>
                    </div>
                    <div style='margin-top: 10px; display: flex; align-items: baseline; gap: 8px;'>
                        <span style='font-size: 46px; font-weight: 800; color: #FFFFFF;'>{v_close:,.2f}</span>
                        <span style='font-size: 18px; color: #AAAAAA; font-weight: 600;'>{currency}</span>
                    </div>
                    <div style='margin-top: 5px; font-size: 16px; font-weight: 700; color: {c_color};'>
                        {c_sign}{day_change:,.2f} ({c_sign}{day_change_pct:.2f}%) {c_arrow} 변동
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # =========================
            # 핵심 AI 연산 및 하이브리드 가중치 부여 (기존 엔진 복구)
            # =========================
            ai_close_tuple = tuple(df_ai["Close"].dropna().to_numpy().flatten())
            steps = 30
            
            with st.spinner("기존 AI 딥러닝 엔진 구동 중 (ARIMA / LSTM / Transformer)..."):
                pred_arima = forecast_arima_cached(ai_close_tuple, steps=steps)
                pred_lstm = forecast_lstm_cached(ai_close_tuple, steps=steps)
                pred_trans = forecast_transformer_cached(ai_close_tuple, steps=steps)
                
                # 앙상블 기본 베이스 예측치 생성
                ensemble_base = (pred_arima * 0.2) + (pred_lstm * 0.4) + (pred_trans * 0.4)
                
                # 기존 모멘텀 분석 점수 추출
                pred_momentum, momentum_score, metrics_dict = make_momentum_ai_forecast(df_ai, ensemble_base, steps=steps)
                
            # 🚀 [뉴스 감성 및 배당 시즌 역사적 가중치 엔진 레이어 결합]
            with st.spinner("하이브리드 이벤트-오버레이 가중치 연산 중..."):
                current_month = datetime.now().month
                is_dividend_season = current_month in [11, 12]
                
                # 가중치 바이어스 산출
                sentiment_bias = sentiment_score * 0.015  # 긍정 뉴스 가중치
                dividend_bias = 0.008 if (is_dividend_season and is_korea) else 0.0  # 전통적인 연말 배당향 가중치
                
                total_bias = sentiment_bias + dividend_bias
                
                # 최종 하이브리드 예측선 생성 (기존 모멘텀 예측을 기반으로 이벤트 오버레이)
                hybrid_predictions = []
                prev_p = v_close
                for i in range(steps):
                    decay = 0.95 ** i  # 미래로 갈수록 당일 뉴스의 영향력 감쇠
                    base_return = (pred_momentum[i] - prev_p) / prev_p if prev_p > 0 else 0
                    adjusted_return = base_return + (total_bias * decay)
                    
                    next_p = prev_p * (1 + adjusted_return)
                    hybrid_predictions.append(next_p)
                    prev_p = next_p
                hybrid_predictions = np.array(hybrid_predictions)

            # =========================
            # 차트 구성
            # =========================
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28]
            )

            # 과거 차트 로직
            if is_daily_chart:
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                    name="주가", increasing_line_color="#FF3232", decreasing_line_color="#0062FF"
                ), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["MA5"], line=dict(color="#00FF00", width=1.5), name="5일선"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], line=dict(color="#FF0000", width=1.5), name="20일선"), row=1, col=1)
                if selected_period_label in ["6개월", "1년", "5년"]:
                    fig.add_trace(go.Scatter(x=df.index, y=df["MA60"], line=dict(color="#FF9900", width=1.5), name="60일선"), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color="#FFFFFF", name="거래량", showlegend=False), row=2, col=1)
                fig.update_xaxes(type="date", rangebreaks=[dict(bounds=["sat", "mon"])], row=1, col=1)
                fig.update_xaxes(type="date", rangebreaks=[dict(bounds=["sat", "mon"])], row=2, col=1)
                y_pad = (v_high - v_low) * 0.05 if (v_high - v_low) > 0 else 1
                fig.update_yaxes(range=[v_low - y_pad, v_high + y_pad], row=1, col=1)
            else:
                line_color = "#FF3232" if period_return >= 0 else "#0062FF"
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["Close"], mode="lines", line=dict(color=line_color, width=2.5),
                    fill="tozeroy", fillcolor="rgba(255, 50, 50, 0.08)" if period_return >= 0 else "rgba(0, 98, 255, 0.08)",
                    name="주가"
                ), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color="#444444", name="거래량", showlegend=False), row=2, col=1)
                latest_date = df.index[-1]
                x_start = latest_date.replace(hour=9, minute=0 if is_korea else 30, second=0)
                x_end = latest_date.replace(hour=15 if is_korea else 16, minute=30 if is_korea else 0, second=0)
                fig.update_xaxes(type="date", range=[x_start, x_end], tickformat="%H:%M", row=1, col=1)
                fig.update_xaxes(type="date", range=[x_start, x_end], tickformat="%H:%M", row=2, col=1)
                y_margin = (v_high - v_low) * 0.08 if (v_high - v_low) > 0 else v_close * 0.002
                fig.update_yaxes(range=[v_low - y_margin, v_high + y_margin], autorange=False, row=1, col=1)

            # 🚀 [미래 차트 연장 플로팅]
            last_date = df.index[-1]
            future_dates = []
            tmp_date = last_date
            while len(future_dates) < steps:
                tmp_date += timedelta(days=1)
                if tmp_date.weekday() < 5:  # 주말 제외
                    future_dates.append(tmp_date)

            # 기존 3가지 예측 모델선 시각화 복구
            fig.add_trace(go.Scatter(x=future_dates, y=pred_arima, line=dict(color="#888888", width=1.5, dash="dash"), name="ARIMA 예측"), row=1, col=1)
            fig.add_trace(go.Scatter(x=future_dates, y=pred_lstm, line=dict(color="#FFAAAA", width=1.5, dash="dash"), name="LSTM 예측"), row=1, col=1)
            fig.add_trace(go.Scatter(x=future_dates, y=pred_trans, line=dict(color="#AAFFAA", width=1.5, dash="dash"), name="Transformer 예측"), row=1, col=1)
            fig.add_trace(go.Scatter(x=future_dates, y=pred_momentum, line=dict(color="#FF00FF", width=2), name="모멘텀 추세선"), row=1, col=1)
            
            # 신규 하이브리드 가중치 오버레이 결과선 레이어링
            fig.add_trace(go.Scatter(
                x=future_dates, y=hybrid_predictions, 
                line=dict(color="#00FFCC", width=3), 
                name="★ 하이브리드 가중치 모델 (뉴스/배당 반영)"
            ), row=1, col=1)

            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#000000", height=550,
                xaxis_rangeslider_visible=False, margin=dict(l=10, r=60, t=10, b=10),
                legend=dict(x=0.01, y=0.99, orientation="h", font=dict(color="white")),
                xaxis=dict(showgrid=True, gridcolor="#222222", linecolor="#FFFFFF"),
                yaxis=dict(showgrid=True, gridcolor="#222222", linecolor="#FFFFFF", side="right", tickformat=",.2f")
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # 크롤링된 실시간 경제 뉴스 컴포넌트 하단 렌더링
            if news_items:
                st.markdown(f"### 📰 {comp_full_name} 최근 경제 뉴스 (실시간 반영점수: {sentiment_score:+.2f})")
                for item in news_items:
                    st.markdown(f"""
                        <div class="news-card">
                            <a class="news-title" href="{item['link']}" target="_blank">{item['title']}</a>
                            <div class="news-meta">{item['source']} | {item['date']}</div>
                            <div class="news-summary-box">{item['desc']}</div>
                        </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
