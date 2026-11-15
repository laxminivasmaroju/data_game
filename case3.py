import numpy as np, plotly.graph_objects as go, streamlit as st

def _init():
    st.session_state.setdefault("credibility", 100)
    st.session_state.setdefault("solved", set())

def _adj(d):
    st.session_state["credibility"] = int(max(0, min(100, st.session_state["credibility"] + d)))

def _clue(cid, label, ok, hint):
    key = "clue:" + cid
    if key in st.session_state["solved"]:
        st.success("📌 Clue saved: " + label); return
    if st.button("📌 Save clue: " + label, key="btn_" + key):
        if ok:
            st.session_state["solved"].add(key); _adj(5); st.success("Clue verified! +5% Credibility")
        else:
            st.warning(hint)

def _f(t):
    u = (t - 20) / 10
    return 300 + 60 * u + 40 * u ** 2 + 25 * u ** 3

@st.cache_data
def make_data():
    rng = np.random.default_rng(3)
    t_tr = np.sort(rng.uniform(8, 32, 40)); t_te = np.sort(rng.uniform(33, 40, 15))
    return t_tr, _f(t_tr) + rng.normal(0, 18, 40), t_te, _f(t_te) + rng.normal(0, 18, 15)

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def run_case3():
    _init()
    st.title("Case 003: The Power Surge")
    st.info("**Learning Objectives:** fit polynomial regression, see underfitting vs overfitting, and balance "
            "bias and variance by judging a model on unseen data.")
    st.write("A heatwave is coming. Predict electricity load (MW) from temperature (°C) using past data, "
             "then deploy your model to the city grid. The heatwave data (33–40 °C) is **hidden** until you deploy.")
    t_tr, y_tr, t_te, y_te = make_data()
    deg = st.slider("Polynomial degree", 1, 12, 1)
    with np.errstate(all="ignore"):
        poly = __import__("numpy").polynomial.Polynomial.fit(t_tr, y_tr, deg)
    tr = rmse(poly(t_tr), y_tr)
    st.metric("Training RMSE", f"{tr:.1f} MW")

    deployed = st.session_state.get("c3_deployed")
    fig = go.Figure()
    fig.add_scatter(x=t_tr, y=y_tr, mode="markers", name="Training data")
    show_test = deployed is not None
    xs = np.linspace(8, 40 if show_test else 32, 300)
    fig.add_scatter(x=xs, y=poly(xs), mode="lines", name=f"Degree {deg} fit")
    if show_test:
        fig.add_scatter(x=t_te, y=y_te, mode="markers", name="Heatwave test data", marker=dict(symbol="x", size=9))
    fig.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Load (MW)", yaxis_range=[0, 1300])
    st.plotly_chart(fig, use_container_width=True)

    _clue("c3_sweet", "A smooth curve that follows the trend, not every noisy point",
          3 <= deg <= 4, "Not verified – the curve either misses the trend or chases noise at this degree.")

    if "case3" in st.session_state["solved"]:
        te = rmse(poly(t_te), y_te)
        st.success(f"✅ Grid stabilised! Train RMSE {tr:.1f} MW vs Test RMSE {te:.1f} MW – similar errors mean good generalisation.")
        st.markdown("**Concept Debrief:** Degree 1 has high *bias* (underfits, misses the surge → blackout). "
                    "Degree 9–12 has high *variance* (memorises noise, ~perfect training fit but wild test error → false shutdown). "
                    "Degree 3–4 balances both. Always judge a model on data it hasn't seen.")
        return
    if st.button("🚀 Deploy Model to City Grid"):
        te = rmse(poly(t_te), y_te)
        st.session_state["c3_deployed"] = deg
        c1, c2 = st.columns(2); c1.metric("Train RMSE", f"{tr:.1f}"); c2.metric("Test RMSE", f"{te:.1f}")
        if deg <= 2:
            _adj(-35); st.error("🔌 BLACKOUT – underfitting: the model missed the peak surge. −35% Credibility")
        elif deg >= 9:
            _adj(-35); st.error("⚠️ FALSE SHUTDOWN – overfitting: great on training data, wild on the heatwave. −35% Credibility")
        elif deg <= 4:
            st.session_state["solved"].add("case3"); st.rerun()
        else:
            st.warning("Grid wobbling – predictions drift in the heatwave. Close, but not stable. No penalty; try a simpler model.")

if __name__ == "__main__":
    _init()
    run_case3()
