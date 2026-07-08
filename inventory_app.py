import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- 数据库初始化 ----------
DB_NAME = "inventory.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            unit TEXT DEFAULT '个',
            min_stock REAL DEFAULT 10,
            max_stock REAL DEFAULT 100
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS inventory_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            type TEXT CHECK(type IN ('in', 'out')) NOT NULL,
            quantity REAL NOT NULL,
            quality TEXT DEFAULT '',
            unit_price REAL DEFAULT 0.0,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------- 数据库操作函数 ----------
def get_products():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM products", conn)
    conn.close()
    return df

def add_product(name, unit, min_stock, max_stock):
    conn = sqlite3.connect(DB_NAME)
    try:
        conn.execute("INSERT INTO products (name, unit, min_stock, max_stock) VALUES (?,?,?,?)",
                     (name, unit, min_stock, max_stock))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

def update_product_thresholds(product_id, min_stock, max_stock):
    conn = sqlite3.connect(DB_NAME)
    conn.execute("UPDATE products SET min_stock=?, max_stock=? WHERE id=?",
                 (min_stock, max_stock, product_id))
    conn.commit()
    conn.close()

def add_log(product_id, log_type, quantity, quality, unit_price):
    conn = sqlite3.connect(DB_NAME)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("INSERT INTO inventory_log (product_id, type, quantity, quality, unit_price, timestamp) VALUES (?,?,?,?,?,?)",
                 (product_id, log_type, quantity, quality, unit_price, now))
    conn.commit()
    conn.close()

def get_current_stock(product_id=None):
    conn = sqlite3.connect(DB_NAME)
    if product_id:
        query = '''
            SELECT product_id,
                   SUM(CASE WHEN type='in' THEN quantity ELSE 0 END) -
                   SUM(CASE WHEN type='out' THEN quantity ELSE 0 END) AS stock
            FROM inventory_log
            WHERE product_id = ?
            GROUP BY product_id
        '''
        result = pd.read_sql(query, conn, params=(product_id,))
    else:
        query = '''
            SELECT product_id,
                   SUM(CASE WHEN type='in' THEN quantity ELSE 0 END) -
                   SUM(CASE WHEN type='out' THEN quantity ELSE 0 END) AS stock
            FROM inventory_log
            GROUP BY product_id
        '''
        result = pd.read_sql(query, conn)
    conn.close()
    return result

def get_inventory_log(product_id, start_date=None, end_date=None):
    conn = sqlite3.connect(DB_NAME)
    query = "SELECT * FROM inventory_log WHERE product_id = ?"
    params = [product_id]
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    query += " ORDER BY timestamp"
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

def get_daily_stock_series(product_id, start_date, end_date):
    """ 构造每日库存时间序列，用于画图 """
    log_df = get_inventory_log(product_id, start_date, end_date)
    if log_df.empty:
        return pd.DataFrame(columns=["date", "stock"])
    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
    log_df = log_df.sort_values("timestamp")
    # 计算累计库存
    cum_stock = 0
    records = []
    for _, row in log_df.iterrows():
        if row["type"] == "in":
            cum_stock += row["quantity"]
        else:
            cum_stock -= row["quantity"]
        records.append({"date": row["timestamp"].date(), "stock": cum_stock})
    daily_df = pd.DataFrame(records)
    # 按天聚合并取最后时刻的库存
    daily_df = daily_df.groupby("date").last().reset_index()
    # 填充无记录日期的库存为前一日值
    if not daily_df.empty:
        full_dates = pd.date_range(start=daily_df["date"].min(), end=daily_df["date"].max(), freq="D")
        daily_df = daily_df.set_index("date").reindex(full_dates).ffill().reset_index()
        daily_df.rename(columns={"index": "date"}, inplace=True)
    return daily_df

# ---------- Streamlit 界面 ----------
st.set_page_config(page_title="库存管理系统", layout="wide")
st.title("📦 智能库存管理系统")

# 侧边栏：商品管理 & 入库/出库
with st.sidebar:
    st.header("➕ 添加新商品")
    with st.form("add_product_form"):
        name = st.text_input("商品名称")
        unit = st.text_input("单位", value="个")
        min_stock = st.number_input("库存下限（紧张提醒）", value=10.0, step=1.0)
        max_stock = st.number_input("库存上限（过多提醒）", value=100.0, step=1.0)
        submitted = st.form_submit_button("添加商品")
        if submitted and name:
            if add_product(name, unit, min_stock, max_stock):
                st.success(f"已添加 {name}")
            else:
                st.error("商品名称重复！")

    st.header("📥 入库 / 📤 出库")
    products_df = get_products()
    if not products_df.empty:
        product_names = products_df["name"].tolist()
        selected_product = st.selectbox("选择商品", product_names)
        product_info = products_df[products_df["name"] == selected_product].iloc[0]
        product_id = int(product_info["id"])

        log_type = st.radio("操作类型", ["入库 (in)", "出库 (out)"])
        quantity = st.number_input("数量", min_value=0.0, step=1.0, value=1.0)
        quality = st.text_input("质量（选填，如优/良/合格）", value="")
        unit_price = st.number_input("单价", min_value=0.0, step=0.1, value=0.0)

        if st.button("确认提交"):
            if quantity <= 0:
                st.error("数量必须大于0")
            else:
                add_log(product_id, "in" if "入" in log_type else "out",
                         quantity, quality, unit_price)
                st.success("记录已保存")

        # 修改当前商品的阈值
        st.subheader("⚙️ 修改库存提醒阈值")
        new_min = st.number_input("新下限", value=float(product_info["min_stock"]), step=1.0, key="min_mod")
        new_max = st.number_input("新上限", value=float(product_info["max_stock"]), step=1.0, key="max_mod")
        if st.button("更新阈值"):
            update_product_thresholds(product_id, new_min, new_max)
            st.success("阈值已更新")
            st.experimental_rerun()
    else:
        st.info("请先添加商品")

# 主界面
tab1, tab2, tab3 = st.tabs(["📊 库存总览 & 提醒", "📈 历史趋势 & 预测", "📋 出入库流水"])

with tab1:
    st.subheader("当前库存状况")
    current_stocks = get_current_stock()
    products_df = get_products()

    if products_df.empty:
        st.info("还没有商品，请先在侧边栏添加。")
    else:
        # 合并库存与商品信息
        merged = products_df.merge(current_stocks, left_on="id", right_on="product_id", how="left")
        merged["stock"] = merged["stock"].fillna(0)
        merged["status"] = "正常"
        merged.loc[merged["stock"] < merged["min_stock"], "status"] = "⚠️ 库存紧张"
        merged.loc[merged["stock"] > merged["max_stock"], "status"] = "📈 库存过多"

        st.dataframe(merged[["name", "unit", "stock", "min_stock", "max_stock", "status"]], use_container_width=True)

        # 单独提醒
        alerts = merged[merged["status"] != "正常"]
        if not alerts.empty:
            st.warning("以下商品需要关注：")
            for _, row in alerts.iterrows():
                if "紧张" in row["status"]:
                    st.error(f"{row['name']} 当前库存 {row['stock']}，低于下限 {row['min_stock']}")
                else:
                    st.warning(f"{row['name']} 当前库存 {row['stock']}，高于上限 {row['max_stock']}")

with tab2:
    st.subheader("库存趋势与预测")
    if not products_df.empty:
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_product_trend = st.selectbox("选择商品查看趋势", products_df["name"].tolist(), key="trend_product")
            days_lookback = st.slider("历史数据天数", 7, 90, 30)
            forecast_days = st.slider("预测未来天数", 1, 30, 7)

        product_id_trend = int(products_df[products_df["name"] == selected_product_trend]["id"].iloc[0])
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_lookback)

        daily_stock = get_daily_stock_series(product_id_trend, start_date.isoformat(), end_date.isoformat())

        if daily_stock.empty:
            st.info("该商品在所选时间段内无流水记录。")
        else:
            # 历史趋势图
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_stock["date"], y=daily_stock["stock"],
                mode="lines+markers", name="历史库存", line=dict(color="royalblue")
            ))

            # 预测：使用最近N天的数据进行线性回归
            forecast_df = pd.DataFrame()
            if len(daily_stock) >= 3:
                # 准备数据
                df_model = daily_stock.copy()
                df_model["day_num"] = (df_model["date"] - df_model["date"].min()).dt.days
                X = df_model[["day_num"]].values
                y = df_model["stock"].values

                model = LinearRegression()
                model.fit(X, y)

                # 生成未来日期和预测
                last_day_num = df_model["day_num"].max()
                future_days = np.arange(last_day_num + 1, last_day_num + forecast_days + 1).reshape(-1, 1)
                future_stock = model.predict(future_days)
                future_dates = [df_model["date"].max() + timedelta(days=int(i)) for i in range(1, forecast_days + 1)]

                # 添加预测线
                fig.add_trace(go.Scatter(
                    x=future_dates, y=future_stock,
                    mode="lines+markers", name="预测库存", line=dict(dash="dash", color="firebrick")
                ))

                # 标注当前商品阈值
                product_row = products_df[products_df["id"] == product_id_trend].iloc[0]
                min_lvl = product_row["min_stock"]
                max_lvl = product_row["max_stock"]
                fig.add_hline(y=min_lvl, line_dash="dot", line_color="orange",
                              annotation_text="下限", annotation_position="bottom right")
                fig.add_hline(y=max_lvl, line_dash="dot", line_color="green",
                              annotation_text="上限", annotation_position="top right")

                # 判断预测是否触发提醒
                last_pred = future_stock[-1]
                if last_pred < min_lvl:
                    st.error(f"预测 {forecast_days} 天后库存将降至 {last_pred:.1f}，低于下限 {min_lvl}！")
                elif last_pred > max_lvl:
                    st.warning(f"预测 {forecast_days} 天后库存将增至 {last_pred:.1f}，高于上限 {max_lvl}。")
                else:
                    st.success("预测期内库存处于正常范围。")

            fig.update_layout(
                title=f"{selected_product_trend} 库存趋势与预测",
                xaxis_title="日期",
                yaxis_title="库存量",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 提供历史数据的简单统计
            st.subheader("过去时段统计摘要")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("期初库存", f"{daily_stock['stock'].iloc[0]:.1f}")
            col_b.metric("期末库存", f"{daily_stock['stock'].iloc[-1]:.1f}")
            col_c.metric("平均库存", f"{daily_stock['stock'].mean():.1f}")
    else:
        st.info("请先添加商品。")

with tab3:
    st.subheader("出入库流水记录")
    if not products_df.empty:
        filter_product = st.selectbox("过滤商品（可选）", ["全部"] + products_df["name"].tolist(), key="log_filter")
        if filter_product == "全部":
            conn = sqlite3.connect(DB_NAME)
            log_df = pd.read_sql("""
                SELECT l.id, p.name AS 商品, l.type, l.quantity, l.quality, l.unit_price, l.timestamp
                FROM inventory_log l
                JOIN products p ON l.product_id = p.id
                ORDER BY l.timestamp DESC
            """, conn)
            conn.close()
    else:
            # 加上这一行防崩溃判断：如果因为删光了商品导致 filter_product 未定义，直接提示
            if 'filter_product' not in locals():
                st.info("当前没有商品，请先在侧边栏添加商品后查看流水。")
            else:
                matching_product = products_df[products_df["name"] == filter_product]
                
                if not matching_product.empty:
                    pid = int(matching_product["id"].iloc[0])
                    log_df = get_inventory_log(pid)
                    log_df = log_df.rename(columns={"type": "类型"})
                    log_df["商品"] = filter_product
                    
                    if not log_df.empty:
                        st.dataframe(log_df, use_container_width=True)
                    else:
                        st.info("暂无流水。")
                else:
                    st.warning("⚠️ 当前数据库中没有找到该商品，请先添加库存。")
