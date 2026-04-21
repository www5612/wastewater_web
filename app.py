# 运行方式：streamlit run app.py
import streamlit as st
import numpy as np
import joblib
# ===================== 兼容 numpy 2.0 修复 =====================
import numpy
if not hasattr(numpy.core.numeric, 'ComplexWarning'):
    from numpy.exceptions import ComplexWarning
    numpy.core.numeric.ComplexWarning = ComplexWarning
# ==============================================================

@st.cache_resource
def load_bundle():
    return joblib.load('water_quality3.0.pkl')

bundle = load_bundle()

SELECT_FEATURES  = bundle['select_features']
POLLUTION_COLS   = bundle['pollution_cols']
scaler_feat      = bundle['scaler_feat']
scaler_poll      = bundle['scaler_poll']
trained_models   = bundle['models']
final_weights    = bundle['weights']


# 预测函数：原始输入 → log10(n+1)变换（pH除外）→ 标准化 → 预测 → 反变换

def predict(raw_inputs: dict) -> float:
    """
    raw_inputs: 11个原始指标的字典（原始浓度，未取对数）
    """
    # 1. 对除 pH 外的所有输入值做 log10(n+1) 变换 
    log_inputs = {}
    for key, val in raw_inputs.items():
        if key == 'pH':
            log_inputs[key] = val
        else:
            # 确保非负，取 log10(val+1)
            log_inputs[key] = np.log10(max(0.0, val) + 1.0)

    #2. 构造 BOD/COD（使用变换后的值）
    cod_log = log_inputs.get('化学需氧量（mg/L）', 0.0)
    bod_log = log_inputs.get('生化需氧量（mg/L）', 0.0)
    log_inputs['BOD/COD'] = bod_log / cod_log if cod_log != 0 else 0.0

    # ---------- 3. 提取 7 个特征 
    x_feat = np.array([[log_inputs[f] for f in SELECT_FEATURES]])   # (1,7)

    # ---------- 4. 提取 6 个污染强度列 ----------
    x_poll = np.array([[log_inputs[c] for c in POLLUTION_COLS]])     # (1,6)

    # ---------- 5. 标准化 ----------
    x_feat_scaled = scaler_feat.transform(x_feat)                   # (1,7)
    x_poll_scaled = scaler_poll.transform(x_poll)                   # (1,6)
    intensity     = x_poll_scaled.mean(axis=1, keepdims=True)       # (1,1)

    # ---------- 6. 拼接最终特征 ----------
    x_final = np.hstack([x_feat_scaled, intensity])                 # (1,8)

    # ---------- 7. 加权融合预测（模型输出的是 log10(CFU/L+1)） ----------
    y_pred_log = sum(
        final_weights[name] * model.predict(x_final)[0]
        for name, model in trained_models.items()
    )

    # ---------- 8. 反变换回原始 CFU/L ----------
    y_pred_cfu = 10 ** y_pred_log - 1
    return max(0.0, y_pred_cfu)

# ============================================================
# Streamlit 界面
# ============================================================
st.set_page_config(page_title="废水粪大肠菌群预测", page_icon="💧", layout="centered")
st.title("💧 废水粪大肠菌群浓度预测系统")
st.markdown("输入11项原始水质指标，系统自动预测粪大肠菌群浓度（CFU/L）")
st.divider()

col1, col2 = st.columns(2)

with col1:
    pH    = st.number_input("pH",                    value=7.0,  step=0.1, format="%.2f")
    COD   = st.number_input("化学需氧量（mg/L）",    value=50.0, step=1.0)
    BOD   = st.number_input("生化需氧量（mg/L）",    value=20.0, step=1.0)
    NH3   = st.number_input("氨氮（mg/L）",          value=5.0,  step=0.1)
    TN    = st.number_input("总氮（mg/L）",          value=10.0, step=0.1)
    TP    = st.number_input("总磷（mg/L）",          value=1.0,  step=0.1)

with col2:
    SS    = st.number_input("悬浮物（mg/L）",        value=30.0, step=1.0)
    Color = st.number_input("色度（倍）",            value=10.0, step=1.0)
    Oil   = st.number_input("石油类（mg/L）",        value=0.5,  step=0.1)
    AOil  = st.number_input("动植物油（mg/L）",      value=0.5,  step=0.1)
    LAS   = st.number_input("阴离子表面活性剂（mg/L）", value=0.5, step=0.1)

st.divider()

if st.button("🔍 开始预测", use_container_width=True):
    raw = {
        'pH'              : pH,
        '化学需氧量（mg/L）'    : COD,
        '生化需氧量（mg/L）'    : BOD,
        '氨氮（mg/L）'         : NH3,
        '总氮（mg/L）'         : TN,
        '总磷（mg/L）'         : TP,
        '悬浮物（mg/L）'       : SS,
        '色度（倍）'           : Color,
        '石油类（mg/L）'       : Oil,
        '动植物油（mg/L）'     : AOil,
        '阴离子表面活性剂（mg/L）': LAS,
    }

    result = predict(raw)

    st.subheader("📊 预测结果")
    st.metric(label="粪大肠菌群浓度", value=f"{result:,.0f} CFU/L")

    if result >= 24000:
        st.error(f"⚠️ 严重超标警告！预测浓度 {result:,.0f} CFU/L ≥ 24000 CFU/L，建议立即处理！")
    elif result >= 1000:
        st.warning(f"⚠️ 浓度偏高（{result:,.0f} CFU/L），已经超出《城镇污水处理厂水污染物排放标准》(DB11/890-2012)的排放标准")
    else:
        st.success(f"✅ 浓度正常（{result:,.0f} CFU/L），水质达标。")

    with st.expander("查看中间计算过程"):
        bod_cod_original = BOD / COD if COD != 0 else 0
        st.write(f"原始 BOD/COD = {bod_cod_original:.4f}")
        st.write(f"模型输出的 log10(CFU/L+1) = {np.log10(result + 1):.4f}")
        st.write(f"模型权重：{', '.join(f'{k}={v:.3f}' for k,v in final_weights.items())}")

st.caption("模型：XGBoost + GBDT + Ridge 加权融合 | 预处理：除 pH 外所有特征 log10(n+1) 变换 + 标准化")