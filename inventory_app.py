import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

# ===== 数据库配置 =====
DB_NAME = "inventory.db"

# ===== 初始化数据库 =====
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # 创建商品表
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            unit TEXT DEFAULT '个',
            min_stock REAL DEFAULT 10,
            max_stock REAL DEFAULT 100
        )
    ''')
    # 创建流水日志表
    c.execute('''
        CREATE TABLE IF NOT EXISTS inventory_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            type TEXT CHECK(type IN ('in', 'out')) NOT NULL,
            quantity REAL NOT NULL,
            quality TEXT DEFAULT '',
            unit_price REAL DEFAULT 0.0,
            operator TEXT DEFAULT '',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(product_id) REFERENCES products(id)
        )
    ''')
    conn.commit()
    conn.close()

# 页面启动时初始化数据库
init_db()

# ===== 数据读取函数 =====
def load_products():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM products", conn)
    conn.close()
    return df

def get_inventory_log(product_id):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("""
        SELECT l.id, p.name AS 商品, l.type, l.quantity, l.quality, l.unit_price, l.operator, l.timestamp
        FROM inventory_log l
        JOIN products p ON l.product_id = p.id
        WHERE l.product_id = ?
        ORDER BY l.timestamp DESC
    """, conn, params=(product_id,))
    conn.close()
    return df

def get_daily_stock_series(product_id, start_date, end_date):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("""
        SELECT date(timestamp) as date, type, quantity
        FROM inventory_log
        WHERE product_id = ? AND date(timestamp) BETWEEN ? AND ?
        ORDER BY date
    """, conn, params=(product_id, start_date, end_date))
    conn.close()

    if df.empty:
        return pd.DataFrame()
    
    df['net_change'] = df.apply(lambda row: row['quantity'] if row['type'] == 'in' else -row['quantity'], axis=1)
    daily_net = df.groupby('date')['net_change'].sum().reset_index()
    
    full_dates = pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d').tolist()
    daily_net = daily_net.set_index('date').reindex(full_dates).fillna(0)
    daily_stock = daily_net.cumsum()
    daily_stock = daily_stock.reset_index()
    daily_stock.columns = ['date', 'stock']
    
    return daily_stock

# ===== Streamlit 页面配置与登录验证 =====
st.set_page_config(page_title="智能库存管理系统", layout="wide")

# ---------- 强制登录验证锁 ----------
if "operator_name" not in st.session_state:
    st.session_state.operator_name = ""

st.sidebar.title("📦 智能库存管理")
# 固定显示操作人输入框在最顶部
name_input = st.sidebar.text_input("请输入操作人姓名进行登录", value="")
if st.sidebar.button("确认 / 切换操作人"):
    if name_input.strip() != "":
        st.session_state.operator_name = name_input.strip()
        st.sidebar.success(f"当前操作人已切换为：{st.session_state.operator_name}")
        st.rerun()
    else:
        st.sidebar.error("姓名不能为空！")

# 如果 session 里没有名字，直接中断，不显示下面任何功能
if st.session_state.operator_name == "":
    st.sidebar.info("⚠️ 请先在上方输入姓名并点击确认。")
    st.info("👈 请先在左侧边栏输入您的姓名以开始使用系统。")
    st.stop() # 强制停止符，下面的代码都不会执行
# ------------------------------------------

# ===== 侧边栏：操作区 =====
with st.sidebar:
    st.subheader("➕ 添加新商品")
    with st.form("add_product"):
        p_name = st.text_input("商品名称")
        p_unit = st.text_input("单位", value="个")
        p_min = st.number_input("库存下限 (紧张提醒)", value=10.0)
        p_max = st.number_input("库存上限 (过多提醒)", value=100.0)
        submitted = st.form_submit_button("添加商品")
        if submitted and p_name:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            try:
                c.execute("INSERT INTO products (name, unit, min_stock, max_stock) VALUES (?, ?, ?, ?)", 
                          (p_name, p_unit, p_min, p_max))
                conn.commit()
                st.success(f"商品【{p_name}】添加成功！")
            except sqlite3.IntegrityError:
                st.error(f"商品【{p_name}】已存在！")
            conn.close()
            st.rerun()

    st.divider()
    st.subheader("📤 入库 / 📥 出库")
    products_df = load_products()
    product_names = products_df["name"].tolist()
    if product_names:
        sel_product = st.selectbox("选择商品", product_names)
        op_type = st.radio("操作类型", ["入库 (in)", "出库 (out)"])
        qty = st.number_input("数量", min_value=0.0, value=1.0, step=0.5)
        quality = st.text_input("质量 (选填, 如优/合格)", "")
        unit_price = st.number_input("单价", min_value=0.0, value=0.0, step=0.5)
        
        if st.button("确认操作"):
            p_id = products_df[products_df["name"] == sel_product]["id"].iloc[0]
            op = "in" if "入库" in op_type else "out"
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO inventory_log (product_id, type, quantity, quality, unit_price, operator) VALUES (?, ?, ?, ?, ?, ?)",
                      (p_id, op, qty, quality, unit_price, st.session_state.operator_name))
            conn.commit()
            conn.close()
            st.success(f"操作成功！{sel_product} {op_type} {qty}")
            st.rerun()
    else:
        st.info("暂无商品，请先添加。")

    st.divider()
    st.subheader("🗑️ 删除商品")
    if product_names:
        del_product = st.selectbox("选择要删除的商品", product_names, key="del_sel")
        if st.button("⚠️ 确认永久删除该商品", type="primary"):
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("SELECT id FROM products WHERE name = ?", (del_product,))
            res = c.fetchone()
            if res:
                p_id = res[0]
                try:
                    c.execute("DELETE FROM inventory_log WHERE product_id = ?", (p_id,))
                    c.execute("DELETE FROM products WHERE id = ?", (p_id,))
                    conn.commit()
                    conn.close()
                    st.success(f"商品【{del_product}】已彻底删除！")
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败: {e}")
            else:
                st.warning("未找到该商品！")
    else:
        st.info("暂无商品可删除。")

# ===== 主界面区域 =====
products_df = load_products()

# ===== 核心修复：在内存中计算当前真实库存 =====
if not products_df.empty:
    for idx in range(len(products_df)):
        p_id = products_df.iloc[idx]["id"]
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        # 算出入库总和减去出库总和
        c.execute("SELECT SUM(CASE WHEN type='in' THEN quantity ELSE -quantity END) FROM inventory_log WHERE product_id=?", (p_id,))
        real_stock = c.fetchone()[0]
        conn.close()
        # 如果算出来是 None，就赋予 0.0，否则赋予真实的数字
        products_df.loc[idx, 'stock'] = real_stock if real_stock is not None else 0.0

st.title("📦 智能库存管理系统")
st.caption(f"当前操作人：{st.session_state.operator_name}")

tab1, tab2, tab3 = st.tabs(["📊 库存总览 & 提醒", "📈 历史趋势 & 预测", "📋 出入库流水"])

with tab1:
    st.subheader("当前库存状况")
    if not products_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总商品种类", len(products_df))
        with col2:
            total_val = products_df["max_stock"].sum()
            st.metric("设定总库存上限", f"{total_val:.0f}")

        st.dataframe(products_df, use_container_width=True)
        
        st.subheader("以下商品需要关注：")
        # 此时 products_df 里已经有 'stock' 列了，绝对不会报错
        alert_df = products_df[
            (products_df["stock"] < products_df["min_stock"]) | 
            (products_df["stock"] > products_df["max_stock"])
        ].copy()
        if not alert_df.empty:
            for _, row in alert_df.iterrows():
                if row["stock"] < row["min_stock"]:
                    st.error(f"🟥 {row['name']} 当前库存 {row['stock']}，低于下限 {row['min_stock']}。")
                else:
                    st.warning(f"🟨 {row['name']} 当前库存 {row['stock']}，高于上限 {row['max_stock']}。")
        else:
            st.success("✅ 当前所有商品库存健康！")
    else:
        st.info("还没有商品，请在侧边栏添加。")

with tab2:
    st.subheader("过去时段统计摘要")
    if not products_df.empty:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("商品种类", len(products_df))
        col_b.metric("总库存量", f"{products_df['stock'].sum():.0f}")
        col_c.metric("平均库存", f"{products_df['stock'].mean():.0f}")
        
        st.subheader("📈 库存趋势与预测")
        selected_product_trend = st.selectbox("选择要分析的商品", products_df["name"].tolist())
        p_id_trend = products_df[products_df["name"] == selected_product_trend]["id"].iloc[0]
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        daily_stock = get_daily_stock_series(p_id_trend, start_date, end_date)
        
        if not daily_stock.empty:
            fig = px.line(daily_stock, x='date', y='stock', 
                          title=f"{selected_product_trend} 库存趋势与预测",
                          labels={'date':'日期', 'stock':'库存量'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("该商品在最近 30 天暂无库存变动数据。")
    else:
        st.info("暂无商品数据进行分析。")

with tab3:
    st.subheader("📋 出入库流水记录")
    if not products_df.empty:
        filter_product = st.selectbox("过滤商品 (可选)", ["全部"] + products_df["name"].tolist(), key="log_filter")
        
        if filter_product != "全部":
            conn = sqlite3.connect(DB_NAME)
            log_df = pd.read_sql_query("""
                SELECT l.id, p.name, l.type, l.quantity, l.quality, l.unit_price, l.operator, l.timestamp
                FROM inventory_log l
                JOIN products p ON l.product_id = p.id
                WHERE p.name = ?
                ORDER BY l.timestamp DESC
            """, conn, params=(filter_product,))
            conn.close()
            
            if not log_df.empty:
                log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
                log_df['日期'] = log_df['timestamp'].dt.strftime('%Y-%m-%d')
                log_df['时间'] = log_df['timestamp'].dt.strftime('%H:%M:%S')
                log_df['年份'] = log_df['timestamp'].dt.year
                log_df = log_df.drop(columns=['timestamp'])
                log_df = log_df.rename(columns={"type": "类型", "operator": "操作人"})
                st.dataframe(log_df, use_container_width=True)
            else:
                st.info("暂无该商品流水。")
        else:
            conn = sqlite3.connect(DB_NAME)
            log_df = pd.read_sql_query("""
                SELECT l.id, p.name AS 商品, l.type, l.quantity, l.quality, l.unit_price, l.operator, l.timestamp
                FROM inventory_log l
                JOIN products p ON l.product_id = p.id
                ORDER BY l.timestamp DESC
            """, conn)
            conn.close()
            
            if not log_df.empty:
                log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
                log_df['日期'] = log_df['timestamp'].dt.strftime('%Y-%m-%d')
                log_df['时间'] = log_df['timestamp'].dt.strftime('%H:%M:%S')
                log_df['年份'] = log_df['timestamp'].dt.year
                log_df = log_df.drop(columns=['timestamp'])
                log_df = log_df.rename(columns={"type": "类型", "operator": "操作人"})
                st.dataframe(log_df, use_container_width=True)
            else:
                st.info("系统暂时还没有任何流水记录。")
    else:
        st.info("请先在侧边栏添加商品以产生流水。")
