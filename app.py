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

# 🚀 [새로운 엔진 수입] 무거운 딥러닝 모델 대신 GradientBoosting 도입
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="AI 주식 분석 및 예측 시스템", layout="wide")

# =========================
# 기본 설정
# =========================
np.random.seed(42)

# =========================
# 웹 스타일 (기존과 동일하게 완벽 유지)
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
# 데이터 수집 함수 (기존 유지)
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
    try: f_info = dict(t.fast_info)
    except Exception: f_info = {}
    try: i_info = dict(t.info)
    except Exception: i_info = {}
    return f_info, i_info

@st.cache_data(ttl=300)
def get_live_indices():
    # (기존 코드와 동일하게 생략 없이 동작합니다)
    indices_data = []
    tickers = {"📊 KOSPI 지수": "^KS11", "📈 NASDAQ 100": "^NDX", "🪙 BITCOIN (BTC)": "BTC-USD"}
    for name, ticker in tickers.items():
        try:
            t_data = yf.download(ticker, period="2d", interval="1d", progress=False, auto_adjust=False)
            if len(t_data) >= 2:
                closes = t_data["Close"].to_numpy().flatten()
                current, prev = float(closes[-1]), float(closes[-2])
                change_pct = ((current - prev) / prev) * 100
                is_down = change_pct < 0
                arrow = "▼" if is_down else "▲"
                sign = "" if is_down else "+"
                indices_data.append((name, f"{current:,.2f}", f"{sign}{change_pct:.2f}% {arrow}", is_down))
            else: indices_data.append((name, "로딩중", "-", False))
        except Exception: indices_data.append((name, "N/A", "-", False))
    return indices_data

def search_stock_suggestions(query):
    # (기존 코드와 동일)
    clean_query = query.strip()
    if not clean_query: return []
    korean_market_dict = {
        "삼성전자": "005930.KS", "하이닉스": "000660.KS", "현대차": "005380.KS", 
        "네이버": "035420.KS", "카카오": "035720.KS", "에코프로비엠": "247540.KQ",
        "애플": "AAPL", "테슬라": "TSLA", "엔비디아": "NVDA", "비트코인": "BTC-USD"
    }
    suggestions = []
    lower_query = clean_query.lower()
    for kor_key, ticker_val in korean_market_dict.items():
        if kor_key in lower_query or lower_query in kor_key:
            suggestions.append(f"{kor_key} ({ticker_val})")
    suggestions = list(dict.fromkeys(suggestions))
    try:
        search_obj = yf.Search(clean_query, max_results=5)
        search_res = getattr(search_obj, "tickers", getattr(search_obj, "quotes", []))
        for item in search_res:
            if isinstance(item, dict) and item.get("symbol"):
                option_str = f"{item.get('shortname', item.get('name', '주식'))} ({item.get('symbol')})"
                if option_str not in suggestions: suggestions.append(option_str)
    except: pass
    if not suggestions: suggestions.append(clean_query.upper())
    return suggestions

@st.cache_data(ttl=900)
def get_korean_real_news(keyword):
    # (기존 뉴스 크롤러 유지)
    encoded_keyword = urllib.parse.quote(keyword)
    url = f"https://news.google.com/rss/search?q={encoded_keyword}&hl=ko&gl=KR&ceid=KR:ko"
    news_items = []
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.text)
        for item in root.findall(".//item"):
            title = item.find("title").text or "제목 없음"
            link = item.find("link").text or "#"
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else "날짜 없음"
            raw_desc = item.find("description").text if item.find("description") is not None else ""
            clean_desc = re.sub(r"<[^>]+>", "", raw_desc).strip()
            source = "금융 경제 뉴스"
            if " - " in title: title, source = title.rsplit(" - ", 1)
            news_items.append({"title": title, "link": link, "date": pub_date, "source": source, "desc": clean_desc})
            if len(news_items) >= 3: break
    except: pass
    return news_items

# =========================
# 🛠️ [엔진 교체] 하이브리드 이벤트-오버레이 모델 유틸 함수
# =========================
def calculate_sentiment_score(news_items):
    """실시간 뉴스를 분석하여 긍정/부정 감성 점수(-1.0 ~ 1.0)를 추출합니다."""
    if not news_items: return 0.0
    pos_words = ['호실적', '최고', '상승', '흑자', '계약', '인수', '성장', '돌파', '급등', '호재', '추천', '상회', '기대']
    neg_words = ['적자', '하락', '감소', '과징금', '소송', '리콜', '급락', '악재', '우려', '쇼크', '하회', '부진', '위기']
    
    score = 0
    for item in news_items:
        text = item['title'] + " " + item['desc']
        for w in pos_words:
            if w in text: score += 1
        for w in neg_words:
            if w in text: score -= 1
            
    # 정규화
    final_score = score / (len(news_items) * 2) 
    return max(min(final_score, 1.0), -1.0)

@st.cache_data(show_spinner=False)
def forecast_hybrid_event_overlay(df_hist, current_sentiment, steps=30):
    """머신러닝(GBM) 베이스에 역사적 이벤트 가중치를 덧씌우는 하이브리드 예측"""
    data = df_hist.copy()
    data = data.dropna(subset=['Close'])
    if len(data) < 20: 
        return np.repeat(data['Close'].iloc[-1], steps)

    # 파생 변수 생성
    data['Is_Dividend_Season'] = data.index.month.isin([11, 12]).astype(int)
    # 역사적 감성은 당일 주가 등락으로 대리(Proxy) 계산
    data['Historical_Sentiment'] = data['Close'].pct_change().apply(lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0))
    data['Target'] = data['Close'].shift(-1)
    data['Return'] = (data['Target'] - data['Close']) / data['Close']
    
    df_model = data.dropna()
    
    features = ['Close', 'Volume', 'Is_Dividend_Season', 'Historical_Sentiment']
    X = df_model[features]
    y = df_model['Target']
    
    # 1. GBM 기본 모델 학습
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # 2. 오버레이를 위한 역사적 가중치 계산
    avg_pos_return = df_model[df_model['Historical_Sentiment'] == 1]['Return'].mean()
    avg_neg_return = df_model[df_model['Historical_Sentiment'] == -1]['Return'].mean()
    avg_div_return = df_model[df_model['Is_Dividend_Season'] == 1]['Return'].mean()
    
    avg_pos_return = avg_pos_return if not pd.isna(avg_pos_return) else 0.005
    avg_neg_return = avg_neg_return if not pd.isna(avg_neg_return) else -0.005
    avg_div_return = avg_div_return if not pd.isna(avg_div_return) else 0.002
    
    # 3. 현재 상황에 따른 프리미엄 수치 산출
    current_div = 1 if datetime.now().month in [11, 12] else 0
    event_premium = 0.0
    
    if current_sentiment > 0.1: event_premium += avg_pos_return
    elif current_sentiment < -0.1: event_premium += avg_neg_return
    if current_div == 1: event_premium += avg_div_return
        
    # 4. 미래 예측 시뮬레이션
    preds = []
    curr_price = float(df_model['Close'].iloc[-1])
    curr_vol = float(df_model['Volume'].mean()) # 거래량은 평균치 가정
    
    for _ in range(steps):
        next_features = np.array([[curr_price, curr_vol, current_div, current_sentiment]])
        base_pred = model.predict(next_features)[0]
        
        # 가중치 오버레이 적용
        final_pred = base_pred * (1 + event_premium)
        
        # 너무 멀어질수록 이벤트 영향력이 감소하도록 세팅
        event_premium *= 0.85 
        
        preds.append(final_pred)
        curr_price = final_pred
        
    return np.array(preds)


# =========================
# 헤더 (기존 유지)
# =========================
st.markdown("""
    <div class="search-container">
        <h1 style='text-align: center; font-size: 36px; margin-bottom: 10px;'>
            📈 <span style='color: #4F46E5;'>PRO</span> AI CHART ANALYSIS SYSTEM
        </h1>
        <p style='text-align: center; color: #AAAAAA; font-size: 15px;'>Hybrid Event-Overlay (뉴스 감성 & 배당 시즌 반영) 모델 기반 미래 주가 예측</p>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.2, 2, 1.2])
user_input = None
company_clean_name = "삼성전자"

with col2:
    search_query = st.text_input("주식 검색창", placeholder="기업 이름 또는 티커 입력 (예: 삼성전자, Apple, TSLA)", label_visibility="collapsed")
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
        st.markdown("<p class='sub-text'>기업 이름이나 티커를 입력하면 연관 주식 선택지가 아래에 나타납니다.</p>", unsafe_allow_html=True)


# =========================
# 메인 화면
# =========================
if user_input:
    period_map = {"1일": "1d", "1개월": "1mo", "6개월": "6mo", "1년": "1y", "5년": "5y"}
    interval_map = {"1일": "1m", "1개월": "1d", "6개월": "1d", "1년": "1d", "5년": "1d"}

    selected_period_label = st.radio(" 📊 차트 조회 기간 선택", list(period_map.keys()), horizontal=True)
    is_daily_chart = selected_period_label != "1일"

    try:
        with st.spinner("차트 데이터 동기화 중..."):
            p_code = "5y" if is_daily_chart else period_map[selected_period_label]
            i_code = interval_map[selected_period_label]

            df_raw = fetch_stock_data(user_input, p_code, i_code)
            df_ai = fetch_stock_data(user_input, "2y", "1d") # AI 모델 학습용 2년치 데이터
            fast_info, info_dict = fetch_ticker_info(user_input)
            news_data = get_korean_real_news(company_clean_name) # 뉴스 데이터 로드
            current_sentiment = calculate_sentiment_score(news_data) # 실시간 감성 점수 계산

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
                    if isinstance(df[col], pd.DataFrame): df[col] = df[col].iloc[:, 0]
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["Open", "High", "Low", "Close"])

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
            
            # 정보 그리드 (기존 유지)
            st.markdown(f"""
                <div class="info-grid">
                    <div class="info-card"><div class="info-label">시가 Open</div><div class="info-value">{v_open:,.2f} {currency}</div></div>
                    <div class="info-card"><div class="info-label">최고가 High</div><div class="info-value" style="color: #FF3232 !important;">{v_high:,.2f} {currency}</div></div>
                    <div class="info-card"><div class="info-label">최저가 Low</div><div class="info-value" style="color: #0062FF !important;">{v_low:,.2f} {currency}</div></div>
                    <div class="info-card"><div class="info-label">뉴스 감성점수</div><div class="info-value">{'호재🔥' if current_sentiment>0 else '악재❄️' if current_sentiment<0 else '평이'}</div></div>
                    <div class="info-card"><div class="info-label">배당시즌</div><div class="info-value">{'진행중' if datetime.now().month in [11,12] else '아님'}</div></div>
                    <div class="info-card"><div class="info-label">{selected_period_label} 수익률</div><div class="info-value">{return_sign}{period_return:.2f}%</div></div>
                </div>
            """, unsafe_allow_html=True)

            comp_full_name = info_dict.get("longName", company_clean_name)

            if len(df) >= 2:
                close_arr = df["Close"].to_numpy().flatten()
                day_change = float(close_arr[-1] - close_arr[-2])
                day_change_pct = float((day_change / close_arr[-2]) * 100)
            else:
                day_change, day_change_pct = 0.0, 0.0

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
            # 하이브리드 엔진 예측 수행 (기존 코드 뒷단 연결부)
            # =========================
            with st.spinner("AI가 뉴스 감성과 배당 조건을 기반으로 미래 30일을 예측하고 있습니다..."):
                future_steps = 30
                # 앞서 생성해둔 하이브리드 함수 호출!
                hybrid_predictions = forecast_hybrid_event_overlay(df_ai, current_sentiment, steps=future_steps)
                
                # 미래 날짜 생성 (휴일 제외)
                last_date = df.index[-1]
                future_dates = []
                current_date = last_date
                while len(future_dates) < future_steps:
                    current_date += timedelta(days=1)
                    if current_date.weekday() < 5:  # 주말 제외
                        future_dates.append(current_date)

            # =========================
            # 차트 그리기
            # =========================
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28])

            # 과거 주가 캔들스틱
            if is_daily_chart:
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                    name="과거 주가", increasing_line_color="#FF3232", decreasing_line_color="#0062FF"
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", line=dict(color="#FF3232", width=2.5), name="주가"), row=1, col=1)

            # 거래량
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color="#444444", name="거래량", showlegend=False), row=2, col=1)

            # 🚀 [새로운 AI 예측선 추가]
            fig.add_trace(go.Scatter(
                x=future_dates, 
                y=hybrid_predictions, 
                mode="lines+markers", 
                line=dict(color="#00FFCC", width=3, dash="dot"), 
                marker=dict(size=4, color="#4F46E5"),
                name="AI 하이브리드 예측선"
            ), row=1, col=1)

            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#000000", height=600,
                xaxis_rangeslider_visible=False, margin=dict(l=10, r=60, t=10, b=10),
                legend=dict(x=0.01, y=0.99, orientation="h", font=dict(color="white"))
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # 뉴스 요약 섹션 (기존 기능 유지)
            if news_data:
                st.markdown("<h3 style='margin-top:30px;'>📰 최근 관련 뉴스 (AI 감성 분석용)</h3>", unsafe_allow_html=True)
                for news in news_data:
                    st.markdown(f"""
                        <div class="news-card">
                            <a href="{news['link']}" target="_blank" class="news-title">{news['title']}</a>
                            <div class="news-meta">{news['source']} | {news['date']}</div>
                            <div class="news-summary-box">{news['desc']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
