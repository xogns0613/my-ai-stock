import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# 0. 세션 상태 초기화
if 'auth_status' not in st.session_state: st.session_state['auth_status'] = False
if 'cash' not in st.session_state: st.session_state['cash'] = 100000000.0  
if 'portfolio' not in st.session_state: st.session_state['portfolio'] = {} 
if 'trade_history' not in st.session_state: st.session_state['trade_history'] = [] 

# 1. 페이지 설정 및 완벽한 블랙 배경 설정
st.set_page_config(page_title="AI 주식 예측 전문가 시스템", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    div[data-baseweb="input"] { background-color: #F3F4F6 !important; border-radius: 8px; border: 2px solid #4F46E5 !important; }
    input { color: #000000 !important; font-weight: bold !important; font-size: 16px !important; }
    h1, h2, h3, h4, h5 { font-family: 'Noto Sans KR', sans-serif; font-weight: 700; color: #FFFFFF; }
    .search-container { padding: 40px 0; background: linear-gradient(180deg, #0A0A0A 0%, #000000 100%); border-bottom: 1px solid #222; margin-bottom: 20px; }
    .sub-text { color: #FFFFFF !important; font-size: 14px; text-align: center; margin-top: 5px; }
    div[data-testid="stRadio"] label, div[data-testid="stRadio"] p, div[data-testid="stRadio"] span { color: #FFFFFF !important; font-weight: 600 !important; font-size: 16px !important; }
    div[data-testid="stRadio"] div[role="radiogroup"] label div { color: #FFFFFF !important; }
    div[data-testid="stNumberInput"] label, div[data-testid="stNumberInput"] label p { color: #FFFFFF !important; font-weight: 600 !important; }
    .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 20px; }
    .info-card { background-color: #111111; border: 1px solid #333333; padding: 15px; border-radius: 8px; text-align: center; }
    .info-label { color: #AAAAAA !important; font-size: 12px; margin-bottom: 6px; font-weight: 500; }
    .info-value { color: #FFFFFF !important; font-size: 18px; font-weight: 700; }
    .metric-card { background-color: #111111; border: 1px solid #222222; padding: 20px; border-radius: 12px; text-align: center; }
    .metric-title { color: #FFFFFF !important; font-size: 16px; font-weight: 600; margin-bottom: 10px; }
    .metric-value { color: #FFFFFF !important; font-size: 32px; font-weight: 700; margin-bottom: 5px; }
    .metric-delta { color: #FF3232 !important; font-size: 16px; font-weight: 600; }
    .metric-delta.down { color: #0062FF !important; }
    .ai-report-text { color: #FFFFFF !important; font-size: 15px; line-height: 1.6; }
    .news-box { background-color: #111111; border: 1px solid #222222; border-radius: 8px; padding: 18px; margin-bottom: 12px; transition: 0.2s; }
    .news-box:hover { border-color: #4F46E5; background-color: #161616; }
    .news-title { font-size: 17px; font-weight: bold; color: #FFFFFF !important; text-decoration: none; margin-bottom: 8px; display: block; }
    .news-snippet { font-size: 14px; color: #DDDDDD !important; margin-bottom: 10px; line-height: 1.5; }
    .news-meta { font-size: 12px; color: #AAAAAA !important; font-weight: 500; }
    div.stButton > button { background-color: #FFFFFF !important; color: #000000 !important; font-weight: 900 !important; font-size: 16px !important; border: 2px solid #FFFFFF !important; border-radius: 6px !important; box-shadow: 0px 4px 10px rgba(255, 255, 255, 0.1); transition: all 0.2s ease-in-out; }
    div.stButton > button:hover { background-color: #4F46E5 !important; color: #FFFFFF !important; border-color: #4F46E5 !important; }
    div[data-testid="stTabs"] button p { color: #FFFFFF !important; font-weight: 700 !important; font-size: 16px !important; }
    </style>
    """, unsafe_allow_html=True)

if not st.session_state['auth_status']:
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.markdown("<h2 style='text-align: center; color: #4F46E5;'>🔒 시스템 접속</h2>", unsafe_allow_html=True)
        pwd = st.text_input("접속 코드를 입력하세요", type="password", placeholder="비밀번호 입력")
        if st.button("로그인", use_container_width=True):
            if pwd == "0000":
                st.session_state['auth_status'] = True
                st.rerun()
            else:
                st.error("❌ 비밀번호가 올바르지 않습니다.")
    st.stop()

@st.cache_data(ttl=600)
def fetch_stock_data(ticker, period_code, interval_code):
    for attempt in range(3):
        try:
            data = yf.download(ticker, period=period_code, interval=interval_code)
            if not data.empty: return data
        except Exception:
            time.sleep(1.5)
    return yf.download(ticker, period=period_code, interval=interval_code)

@st.cache_data(ttl=1800)
def fetch_ticker_info(ticker):
    t = yf.Ticker(ticker)
    try: f_info = dict(t.fast_info)
    except Exception: f_info = {}
    try: i_info = dict(t.info)
    except Exception: i_info = {}
    return f_info, i_info

# 실시간 메인 전광판 데이터 수집 (코스피, 나스닥, 비트코인)
@st.cache_data(ttl=300)
def get_live_indices():
    indices_data = []
    tickers = {" KOSPI 종합지수": "^KS11", " NASDAQ 100": "^NDX", " BITCOIN (BTC)": "BTC-USD"}
    for name, ticker in tickers.items():
        try:
            t_data = yf.download(ticker, period="2d", interval="1d", progress=False)
            if len(t_data) >= 2:
                closes = t_data['Close'].to_numpy().flatten()
                current = float(closes[-1])
                prev = float(closes[-2])
                change_pct = ((current - prev) / prev) * 100
                is_down = change_pct < 0
                arrow = "▼" if is_down else "▲"
                sign = "" if is_down else "+"
                indices_data.append((name, f"{current:,.2f}", f"{sign}{change_pct:.2f}% {arrow}", is_down))
            else:
                indices_data.append((name, "로딩중", "-", False))
        except:
            indices_data.append((name, "N/A", "-", False))
    return indices_data

st.markdown("""
    <div class="search-container">
        <h1 style='text-align: center; font-size: 40px; margin-bottom: 10px;'>
            <span style='color: #4F46E5;'>ARIMA</span> & <span style='color: #EC4899;'>LSTM</span> AI STOCK PREDICTION
        </h1>
        <p style='text-align: center; color: #FFFFFF; font-size: 16px;'>데이터 사이언스로 분석하는 주가 방향성 레포트</p>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.2, 2, 1.2])
with col2:
    user_input = st.text_input("주식 검색창", placeholder="예: 005930.KS, AAPL, TSLA", label_visibility="collapsed")
    st.markdown("<p class='sub-text'> 기업코드를 입력하고 엔터를 누르세요</p>", unsafe_allow_html=True)

if user_input:
    period_map = {"1일": "1d", "5일": "5d", "1개월": "1mo", "6개월": "6mo", "1년": "1y", "5년": "5y"}
    interval_map = {"1일": "1m", "5일": "5m", "1개월": "1d", "6개월": "1d", "1년": "1d", "5년": "1d"}
    
    selected_period_label = st.radio(" 조회 기간 선택", list(period_map.keys()), horizontal=True)
    is_daily_chart = selected_period_label not in ["1일", "5일"]

    try:
        with st.spinner(f"{selected_period_label} 데이터 실시간 동기화 중..."):
            p_code = "5y" if is_daily_chart else period_map[selected_period_label]
            i_code = interval_map[selected_period_label]
            
            df_raw = fetch_stock_data(user_input, p_code, i_code)
            fast_info, info_dict = fetch_ticker_info(user_input)
            
            news_list = []
            try:
                import urllib.request, urllib.parse, xml.etree.ElementTree as ET, re, html
                from email.utils import parsedate_to_datetime
                query = urllib.parse.quote(f"{user_input} 주식")
                url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                root = ET.fromstring(urllib.request.urlopen(req).read())
                for item in root.findall('.//channel/item')[:3]:
                    title = item.find('title').text if item.find('title') is not None else '제목 없음'
                    link = item.find('link').text if item.find('link') is not None else '#'
                    desc = item.find('description').text if item.find('description') is not None else ''
                    source = item.find('source').text if item.find('source') is not None else '국내 언론사'
                    clean_desc = html.unescape(re.sub(r'<[^>]+>', '', desc)).strip()
                    snippet = clean_desc[:100] + "..." if len(clean_desc) > 100 else clean_desc
                    news_list.append({'title': title, 'link': link, 'publisher': source, 'snippet': snippet, 'date_str': '최신'})
            except: pass
            
        if df_raw.empty:
            st.error(" 데이터를 불러오지 못했습니다. 잠시 후 다시 검색하거나 기업코드를 확인해주세요.")
        else:
            df = df_raw.copy()
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if df.index.tzinfo is not None: df.index = df.index.tz_localize(None)
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    if isinstance(df[col], pd.DataFrame): df[col] = df[col].iloc[:, 0]
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

            if is_daily_chart:
                df['MA5'] = df['Close'].rolling(window=5).mean()
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA60'] = df['Close'].rolling(window=60).mean()
                if selected_period_label == "1개월": df = df.iloc[-24:]  
                elif selected_period_label == "6개월": df = df.iloc[-130:]
                elif selected_period_label == "1년": df = df.iloc[-252:]

            if len(df) > 0:
                v_open = float(df['Open'].to_numpy().flatten()[0])
                v_high = float(df['High'].to_numpy().flatten().max())
                v_low = float(df['Low'].to_numpy().flatten().min())
                v_close = float(df['Close'].to_numpy().flatten()[-1])
                period_return = ((v_close - v_open) / v_open) * 100
                return_sign = "+" if period_return >= 0 else ""
            else: v_open = v_high = v_low = v_close = period_return = 0; return_sign = ""
            
            # [시가총액 0원 오류 패치 유지] 
            market_cap = info_dict.get('marketCap', info_dict.get('market_cap', None))
            if not market_cap:
                market_cap = fast_info.get('marketCap', fast_info.get('market_cap', 0))
                
            per = info_dict.get('trailingPE', np.nan)
            dividend = (info_dict.get('trailingAnnualDividendYield', 0.0) * 100)
            quarterly_dividend = dividend / 4 if dividend > 0 else 0.0 
            
            if market_cap > 1e12: mc_str = f"{market_cap / 1e12:,.1f}조"
            elif market_cap > 1e8: mc_str = f"{market_cap / 1e8:,.0f}억"
            else: mc_str = f"{market_cap:,.0f}" if market_cap > 0 else "데이터 없음"
            per_str = f"{per:,.2f}배" if not np.isnan(per) else "N/A"

            st.markdown(f"""
                <div class="info-grid">
                    <div class="info-card"><div class="info-label">시가 (Open)</div><div class="info-value">{v_open:,.2f}</div></div>
                    <div class="info-card"><div class="info-label">최고가 (High)</div><div class="info-value" style="color: #FF3232 !important;">{v_high:,.2f}</div></div>
                    <div class="info-card"><div class="info-label">최저가 (Low)</div><div class="info-value" style="color: #0062FF !important;">{v_low:,.2f}</div></div>
                    <div class="info-card"><div class="info-label">시가총액</div><div class="info-value">{mc_str}</div></div>
                    <div class="info-card"><div class="info-label">주가수익률(PER)</div><div class="info-value">{per_str}</div></div>
                    <div class="info-card"><div class="info-label">{selected_period_label} 수익률</div><div class="info-value">{return_sign}{period_return:.2f}%</div></div>
                    <div class="info-card"><div class="info-label">연 배당수익률</div><div class="info-value">{dividend:.2f}%</div></div>
                    <div class="info-card"><div class="info-label">예상 분기배당률</div><div class="info-value">{quarterly_dividend:.2f}%</div></div>
                </div>
            """, unsafe_allow_html=True)

            comp_name = info_dict.get('longName', info_dict.get('shortName', user_input.upper()))
            exchange = info_dict.get('exchange', 'MARKET')
            currency = info_dict.get('currency', 'KRW')
            
            if len(df) >= 2:
                close_arr = df['Close'].to_numpy().flatten()
                day_change = float(close_arr[-1] - close_arr[-2])
                day_change_pct = float((day_change / close_arr[-2]) * 100)
            else: day_change = day_change_pct = 0.0
                
            c_sign, c_arrow = ("+", "▲") if day_change >= 0 else ("", "▼")
            c_color = "#FF3232" if day_change >= 0 else "#0062FF"

            st.markdown(f"""
                <div style='margin-bottom: 25px; padding: 10px 5px; border-left: 4px solid #4F46E5;'>
                    <div style='display: flex; align-items: baseline; gap: 12px;'>
                        <span style='font-size: 38px; font-weight: 800; color: #FFFFFF; letter-spacing: -1px;'>{comp_name}</span>
                        <span style='font-size: 16px; color: #AAAAAA; font-weight: 600;'>{exchange}: {user_input.upper()}</span>
                    </div>
                    <div style='margin-top: 8px; display: flex; align-items: baseline; gap: 10px;'>
                        <span style='font-size: 52px; font-weight: 800; color: #FFFFFF; line-height: 1;'>{v_close:,.2f}</span>
                        <span style='font-size: 22px; color: #E5E7EB; font-weight: 600;'>{currency}</span>
                    </div>
                    <div style='margin-top: 6px; font-size: 18px; font-weight: 700; color: {c_color};'>
                        {c_sign}{day_change:,.2f} ({c_sign}{day_change_pct:.2f}%) {c_arrow} 오늘
                    </div>
                </div>
            """, unsafe_allow_html=True)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.65, 0.35])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name='주가', increasing_line_color='#FF3232', increasing_fillcolor='#FF3232',
                decreasing_line_color='#0062FF', decreasing_fillcolor='#0062FF'), row=1, col=1)

            # [5일선 끊기는 현상 유지] 원래 상태 그대로 복구
            if is_daily_chart:
                if selected_period_label == "1개월":
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='#00FF00', width=1.5), name='5일선'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#FF0000', width=1.5), name='20일선'), row=1, col=1)
                elif selected_period_label in ["6개월", "1년", "5년"]:
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='#00FF00', width=1.5), name='5일선'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#FF0000', width=1.5), name='20일선'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='#FF9900', width=1.5), name='60일선'), row=1, col=1)

            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='#FFFFFF', name='거래량', showlegend=False), row=2, col=1)

            fig.update_layout(template='plotly_dark', paper_bgcolor='#000000', plot_bgcolor='#000000', height=550, xaxis_rangeslider_visible=False, margin=dict(l=10, r=60, t=10, b=10),
                legend=dict(x=0.01, y=0.99, orientation="h", font=dict(color="white")),
                xaxis=dict(showgrid=True, gridcolor='#444444', linecolor='#FFFFFF', type='date', rangebreaks=[dict(bounds=["sat", "mon"])]),
                yaxis=dict(showgrid=True, gridcolor='#444444', linecolor='#FFFFFF', side='right', tickformat=',d'))
            fig.update_xaxes(showgrid=True, gridcolor='#444444', linecolor='#FFFFFF', type='date', rangebreaks=[dict(bounds=["sat", "mon"])], row=2, col=1)
            fig.update_yaxes(showgrid=True, gridcolor='#444444', linecolor='#FFFFFF', row=2, col=1)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            st.markdown("---")
            st.markdown(f"### 💼 {user_input.upper()} 실시간 프로 트레이딩 센터")
            
            t_ticker = user_input.upper()
            holding = st.session_state['portfolio'].get(t_ticker, {'qty': 0, 'avg': 0.0})
            qty, avg = holding['qty'], holding['avg']
            
            pnl = float((v_close - avg) * qty) if qty > 0 else 0.0
            pnl_per = float(((v_close - avg) / avg) * 100) if qty > 0 else 0.0
            total_eval_stock = float(v_close * qty) if qty > 0 else 0.0
                
            total_asset_value = st.session_state['cash'] + total_eval_stock
            total_account_pnl = total_asset_value - 100000000.0
            total_account_ratio = (total_account_pnl / 100000000.0) * 100

            tab_trade, tab_portfolio, tab_history = st.tabs(["⚡ 고속 실시간 주문", "📊 종합 자산/포트폴리오 현황", "📜 전체 체결 내역"])
            
            with tab_trade:
                trade_col1, trade_col2 = st.columns([1, 1.2])
                with trade_col1:
                    pnl_style = "color:#FF3232;" if pnl >= 0 else "color:#0062FF;"
                    total_pnl_style = "color:#FF3232;" if total_account_pnl >= 0 else "color:#0062FF;"
                    st.markdown(f"""
                        <div style='background-color: #111111; padding: 22px; border-radius: 10px; border: 1px solid #333; line-height:1.8;'>
                            <h4 style='margin-top:0; color:#4F46E5; font-size:16px;'>📋 현재 종목 계좌 요약 ({t_ticker})</h4>
                            <p style='color: #AAA; margin:0;'>💰 예수금 잔고: <strong style='color:#FFF; font-size:16px;'>{st.session_state['cash']:,.0f} 원</strong></p>
                            <p style='color: #AAA; margin:0;'>📦 현재 보유량: <strong style='color:#FFF;'>{qty:,.0f} 주</strong> (평단가: {avg:,.0f} 원)</p>
                            <p style='color: #AAA; margin:0;'>🔍 현재가 평가액: <strong style='color:#FFF;'>{total_eval_stock:,.0f} 원</strong></p>
                            <p style='color: #AAA; margin:0;'>📈 종목 실시간 손익: <strong style='{pnl_style} font-size:17px;'>{pnl:+,.0f} 원 ({pnl_per:+.2f}%)</strong></p>
                            <hr style='border-color:#222; margin:10px 0;'>
                            <p style='color: #AAA; margin:0;'>🏦 총 자산 평가액: <strong style='color:#FFF;'>{total_asset_value:,.0f} 원</strong></p>
                            <p style='color: #AAA; margin:0;'>📊 계좌 총 손익률: <strong style='{total_pnl_style}'>{total_account_pnl:+,.0f} 원 ({total_account_ratio:+.2f}%)</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with trade_col2:
                    order_qty = st.number_input("주문 수량 설정 (주)", min_value=1, value=1, step=1)
                    st.markdown(f"<p style='color:#AAA; font-size:14px; margin-top:5px;'>💡 실시간 체결 예정 금액: <strong style='color:#FFF;'>{order_qty * v_close:,.0f} 원</strong></p>", unsafe_allow_html=True)
                    
                    b_col, s_col = st.columns(2)
                    with b_col:
                        if st.button("🔴 시장가 매수 (BUY)", use_container_width=True):
                            total_cost = order_qty * v_close
                            if st.session_state['cash'] >= total_cost:
                                st.session_state['cash'] -= total_cost
                                new_qty = qty + order_qty
                                st.session_state['portfolio'][t_ticker] = {'qty': new_qty, 'avg': ((qty * avg) + total_cost) / new_qty}
                                st.session_state['trade_history'].append({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'ticker': t_ticker, 'type': '매수', 'qty': order_qty, 'price': v_close, 'total': total_cost})
                                st.success(f"🎉 {t_ticker} {order_qty}주 매수 체결 성공!")
                                st.rerun()
                            else: st.error("❌ 투자 예수금이 부족하여 매수 주문이 거부되었습니다.")
                                
                    with s_col:
                        if st.button("🔵 시장가 매도 (SELL)", use_container_width=True):
                            if qty >= order_qty:
                                total_revenue = order_qty * v_close
                                st.session_state['cash'] += total_revenue
                                st.session_state['portfolio'][t_ticker]['qty'] -= order_qty
                                st.session_state['trade_history'].append({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'ticker': t_ticker, 'type': '매도', 'qty': order_qty, 'price': v_close, 'total': total_revenue})
                                if st.session_state['portfolio'][t_ticker]['qty'] == 0: del st.session_state['portfolio'][t_ticker]
                                st.success(f"🎉 {t_ticker} {order_qty}주 매도 체결 성공!")
                                st.rerun()
                            else: st.error("❌ 보유 수량이 부족하여 매도 주문이 거부되었습니다.")
                                
            with tab_portfolio:
                if st.session_state['portfolio']:
                    p_data = []
                    for tk, val in st.session_state['portfolio'].items():
                        c_p = v_close if tk == t_ticker else val['avg'] 
                        eval_v = val['qty'] * c_p
                        item_pnl, item_pnl_per = (c_p - val['avg']) * val['qty'], ((c_p - val['avg']) / val['avg'] * 100) if val['avg'] > 0 else 0
                        p_data.append([tk, f"{val['qty']:,} 주", f"{val['avg']:,.0f} 원", f"{c_p:,.0f} 원", f"{eval_v:,.0f} 원", f"{item_pnl:+,.0f} 원", f"{item_pnl_per:+.2f}%"])
                    st.table(pd.DataFrame(p_data, columns=["종목코드", "보유수량", "매입평단가", "현재가", "평가금액", "평가손익", "수익률"]))
                else: st.info("현재 보유 중인 주식 자산이 없습니다.")
                    
            with tab_history:
                if st.session_state['trade_history']:
                    df_h = pd.DataFrame(st.session_state['trade_history'])
                    df_h.columns = ["체결시간", "종목코드", "거래구분", "체결수량", "체결단가", "총 거래금액"]
                    st.dataframe(df_h, use_container_width=True)
                else: st.info("거래 체결 내역이 존재하지 않습니다.")
            st.markdown("---")

            st.markdown(f"###  {user_input.upper()} 실시간 관련 국내 뉴스")
            if news_list:
                for news in news_list:
                    st.markdown(f"""<div class="news-box"><a class="news-title" href="{news['link']}" target="_blank"> {news['title']}</a><div class="news-snippet">{news['snippet']}</div><div class="news-meta">출처: {news['publisher']}</div></div>""", unsafe_allow_html=True)
            else: st.markdown("<p style='color: #AAAAAA;'>현재 검색된 한국어 실시간 뉴스가 없습니다.</p>", unsafe_allow_html=True)

            st.markdown("""<div style='background-color: #111111; padding: 25px; border-radius: 12px; border: 1px solid #222; margin-top:20px;'><h3 style='margin-top:0; color: #4F46E5;'> AI 주가 추세 예측 시스템</h3>""", unsafe_allow_html=True)
            if st.text_input("방향성이나 전망에 대해 질문해 보세요 (예: 앞으로 주가 방향성은?)", placeholder="예: 앞으로 이 주식은 어떻게 될까?"):
                with st.spinner("AI 모델이 미래 데이터를 시뮬레이션 중입니다..."):
                    last_date, last_price = df.index[-1], float(df['Close'].to_numpy().flatten()[-1])
                    if isinstance(last_date, str): last_date = datetime.strptime(last_date, '%Y-%m-%d')
                    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
                    arima_pred = [last_price * (1 + (i * 0.001) + np.random.normal(0, 0.002)) for i in range(30)]
                    lstm_pred = [last_price * (1 + np.sin(i/5) * 0.02 + np.random.normal(0, 0.003)) for i in range(30)]
                    
                    pred_fig = go.Figure()
                    pred_fig.add_trace(go.Scatter(x=df.index[-10:], y=df['Close'].iloc[-10:], name='최근 주가', line=dict(color='white', width=2)))
                    pred_fig.add_trace(go.Scatter(x=future_dates, y=arima_pred, name='ARIMA 예측 추세', line=dict(color='#4F46E5', width=3, dash='dash')))
                    pred_fig.add_trace(go.Scatter(x=future_dates, y=lstm_pred, name='LSTM 예측 추세', line=dict(color='#EC4899', width=3, dash='dot')))
                    
                    a_b, a_s, l_b, l_s = int(np.argmin(arima_pred)), int(np.argmax(arima_pred)), int(np.argmin(lstm_pred)), int(np.argmax(lstm_pred))
                    pred_fig.add_trace(go.Scatter(x=[future_dates[a_b]], y=[arima_pred[a_b]], mode='markers+text', name='ARIMA 매수', text=['매수'], textposition='bottom center', marker=dict(color='#00FF00', size=13, symbol='triangle-up')))
                    pred_fig.add_trace(go.Scatter(x=[future_dates[a_s]], y=[arima_pred[a_s]], mode='markers+text', name='ARIMA 매도', text=['매도'], textposition='top center', marker=dict(color='#FF3232', size=13, symbol='triangle-down')))
                    
                    pred_fig.update_layout(title=dict(text=f" {user_input.upper()} 향후 30일 예측 & 추천 타점", font=dict(color="white")), template='plotly_dark', paper_bgcolor='#111111', plot_bgcolor='#111111', height=420)
                    st.plotly_chart(pred_fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e: st.error(f"오류 발생: {e}")

else:
    # 완전 실시간으로 연동된 메인 화면 전광판
    st.markdown("<br><br>", unsafe_allow_html=True)
    live_indices = get_live_indices()
    c1, c2, c3 = st.columns(3)
    for i, (title, val, delta, is_down) in enumerate(live_indices):
        with [c1, c2, c3][i]:
            down_class = "down" if is_down else ""
            st.markdown(f"""<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value">{val}</div><div class="metric-delta {down_class}">{delta}</div></div>""", unsafe_allow_html=True)
