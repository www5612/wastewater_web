import streamlit as st
import numpy as np
import joblib
import pandas as pd
import io

# ===================== 兼容 numpy 2.0 修复 =====================
import numpy

if not hasattr(numpy.core.numeric, 'ComplexWarning'):
    from numpy.exceptions import ComplexWarning

    numpy.core.numeric.ComplexWarning = ComplexWarning
# ==============================================================

# 设置页面
st.set_page_config(page_title="水质监测 AI 预测系统", page_icon="🧪", layout="wide")

# 自定义 CSS 样式优化 UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e6e9ef; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    # 注意：确保你的 pkl 文件已更新，或者 SELECT_FEATURES 内不包含已删除的指标
    return joblib.load('water_quality3.0.pkl')


try:
    bundle = load_bundle()
    SELECT_FEATURES = bundle['select_features']
    POLLUTION_COLS = bundle['pollution_cols']
    scaler_feat = bundle['scaler_feat']
    scaler_poll = bundle['scaler_poll']
    trained_models = bundle['models']
    final_weights = bundle['weights']
except Exception as e:
    st.error(f"模型加载失败，请检查文件路径: {e}")
    st.stop()


# ===================== 核心预测逻辑 =====================
def predict_core(df_input: pd.DataFrame):
    """
    输入 DataFrame，包含 8 个原始指标：
    pH, COD, BOD, NH3, TN, TP, SS, Color
    """
    df = df_input.copy()

    # 1. log10(n+1) 变换
    for col in df.columns:
        if col != 'pH':
            df[col] = np.log10(df[col].clip(lower=0) + 1.0)

    # 2. 构造 BOD/COD (基于变换后的值)
    # 假设列名严格匹配，若不匹配需在外部映射
    cod_col = '化学需氧量（mg/L）'
    bod_col = '生化需氧量（mg/L）'
    df['BOD/COD'] = df[bod_col] / df[cod_col].replace(0, np.nan)
    df['BOD/COD'] = df['BOD/COD'].fillna(0)

    # 3. 提取特征与标准化
    x_feat = df[SELECT_FEATURES].values
    x_poll = df[POLLUTION_COLS].values

    x_feat_scaled = scaler_feat.transform(x_feat)
    x_poll_scaled = scaler_poll.transform(x_poll)
    intensity = x_poll_scaled.mean(axis=1, keepdims=True)

    x_final = np.hstack([x_feat_scaled, intensity])

    # 4. 集成预测
    y_pred_log = np.zeros(len(df))
    for name, model in trained_models.items():
        y_pred_log += final_weights[name] * model.predict(x_final)

    # 5. 反变换
    y_pred_cfu = 10 ** y_pred_log - 1
    return np.maximum(0, y_pred_cfu)


# ===================== UI 界面 =====================
st.title("🧪 废水粪大肠菌群智能预测系统")
st.caption("基于集成学习算法 (XGBoost + GBDT + Ridge) | 适配精简版 8 项水质指标")

tab1, tab2 = st.tabs(["🎯 单点快速预测", "📂 批量表格预测"])

# ---------------- Tab 1: 单点预测 ----------------
with tab1:
    with st.container():
        st.subheader("输入实时监测数据")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pH = st.number_input("pH值", 0.0, 14.0, 7.0, step=0.1)
            SS = st.number_input("悬浮物 (SS)", 0.0, 5000.0, 30.0)
        with c2:
            COD = st.number_input("化学需氧量 (COD)", 0.0, 10000.0, 50.0)
            TN = st.number_input("总氮 (TN)", 0.0, 500.0, 10.0)
        with c3:
            BOD = st.number_input("生化需氧量 (BOD)", 0.0, 5000.0, 20.0)
            TP = st.number_input("总磷 (TP)", 0.0, 100.0, 1.0)
        with c4:
            NH3 = st.number_input("氨氮 (NH3-N)", 0.0, 500.0, 5.0)
            Color = st.number_input("色度 (Color)", 0.0, 500.0, 10.0)

        if st.button("🚀 开始分析"):
            input_dict = {
                'pH': pH, '化学需氧量（mg/L）': COD, '生化需氧量（mg/L）': BOD,
                '氨氮（mg/L）': NH3, '总氮（mg/L）': TN, '总磷（mg/L）': TP,
                '悬浮物（mg/L）': SS, '色度（倍）': Color
            }
            res = predict_core(pd.DataFrame([input_dict]))[0]

            st.divider()
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.metric("预测浓度 (CFU/L)", f"{res:,.0f}")
            with col_res2:
                if res >= 24000:
                    st.error(f"🚨 严重警告：该样本超标严重，预测值为 {res:,.0f} CFU/L！")
                elif res >= 1000:
                    st.warning(f"⚠️ 注意：已超出排放标准 (DB11/890-2012)。")
                else:
                    st.success("✅ 水质达标：预测浓度在安全范围内。")

# ---------------- Tab 2: 批量预测 ----------------
with tab2:
    st.subheader("上传监测报表 (CSV 或 Excel)")
    uploaded_file = st.file_uploader("将表格拖到此处", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.write("📂 预览上传数据 (前5行):")
            st.dataframe(df_upload.head(), use_container_width=True)

            # 自动列名匹配逻辑（简单示例）
            required_cols = ['pH', '化学需氧量（mg/L）', '生化需氧量（mg/L）', '氨氮（mg/L）',
                             '总氮（mg/L）', '总磷（mg/L）', '悬浮物（mg/L）', '色度（倍）']

            if all(col in df_upload.columns for col in required_cols):
                if st.button("📊 执行批量预测"):
                    with st.spinner('AI 正在深度计算中...'):
                        predictions = predict_core(df_upload[required_cols])
                        df_upload['预测粪大肠菌群(CFU/L)'] = predictions

                    st.success("✅ 计算完成！")
                    st.dataframe(df_upload, use_container_width=True)

                    # 下载按钮
                    towrite = io.BytesIO()
                    df_upload.to_excel(towrite, index=False, engine='openpyxl')
                    st.download_button(
                        label="📥 下载预测结果报告",
                        data=towrite.getvalue(),
                        file_name="预测结果导出.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.error(f"表格格式不正确。请确保包含以下列名：{required_cols}")
        except Exception as e:
            st.error(f"处理文件时出错: {e}")

st.sidebar.title("关于系统")
st.sidebar.info("""
本系统通过对 **8项核心水质指标** 的加权融合模型，实现对粪大肠菌群的高精度估算。
1. **输入说明**：输入原始监测浓度。
2. **算法逻辑**：集成 XGBoost/GBDT和Ridge回归。
3. **指标调整**：已移除油类及表面活性剂指标。
""")
