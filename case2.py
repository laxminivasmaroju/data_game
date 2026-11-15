import numpy as np, pandas as pd, plotly.express as px, streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

@st.cache_data
def make_data():
    rng = np.random.default_rng(11)
    n = 400
    ver = rng.choice(["v2.3", "v2.4"], n, p=[.55, .45])
    tier = rng.choice(["Basic", "Premium"], n, p=[.6, .4])
    bug = (ver == "v2.4") & (tier == "Basic")
    wait = np.where(bug, rng.normal(26, 5, n), rng.normal(8, 3, n)).clip(1, None).round(1)
    churn = np.where(bug, rng.random(n) < .85, rng.random(n) < .06)
    active = np.where(churn, rng.normal(8, 4, n), rng.normal(22, 4, n)).clip(0, 30).round(1)
    df = pd.DataFrame({"support_wait_min": wait, "days_active_30": active, "tenure_months": rng.integers(1, 36, n),
                       "app_version": ver, "plan_tier": tier, "churned": churn.astype(int)})
    df["version_x_tier"] = np.where(bug, "v2.4 · Basic", "Everyone else")
    return df

SUSPECTS = ["Support Lead (slow replies)", "Pricing team (price hike)", "Marketing (bad campaign)",
            "Buggy app v2.4 on Basic-tier devices"]

def run_case2():
    _init()
    st.title("Case 002: The Silent Customers")
    st.info("**Learning Objectives:** tell correlation from causation, find hidden confounders by colouring and "
            "controlling variables, and use K-Means clustering to discover segments.")
    df = make_data()
    r = df.support_wait_min.corr(df.churned)
    st.write("Customers are cancelling. Support wait time and churn correlate strongly – the **Support Lead** is under fire.")
    st.metric("Correlation: support wait vs churn", f"r = {r:.2f}")

    color = st.selectbox("Colour scatter by", ["None", "churned", "app_version", "plan_tier", "version_x_tier"])
    fig = px.scatter(df, x="support_wait_min", y="days_active_30", color=None if color == "None" else df[color].astype(str),
                     opacity=.7, title="Wait time vs days active")
    st.plotly_chart(fig, use_container_width=True)

    ctrl = st.selectbox("Control for (hold constant) →", ["None", "app_version", "plan_tier", "version_x_tier"])
    if ctrl == "None":
        st.write(f"Overall r = {r:.2f}")
    else:
        rows = [{"group": g, "n": len(x), "r(wait, churn)": round(x.support_wait_min.corr(x.churned), 2)}
                for g, x in df.groupby(ctrl)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    k = st.slider("K-Means clusters (k)", 2, 5, 2)
    X = StandardScaler().fit_transform(df[["support_wait_min", "days_active_30", "tenure_months"]])
    d = df.assign(cluster=KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X).astype(str))
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.scatter(d, x="support_wait_min", y="days_active_30", color="cluster", title="K-Means clusters"),
                    use_container_width=True)
    prof = d.groupby("cluster").agg(size=("churned", "size"), avg_wait=("support_wait_min", "mean"),
                                    churn_rate=("churned", "mean"),
                                    pct_v24_basic=("version_x_tier", lambda s: (s == "v2.4 · Basic").mean())).round(2)
    c2.dataframe(prof, use_container_width=True)

    st.subheader("Evidence board")
    _clue("c2_color", "Colouring by version × tier shows one group drives both wait & churn",
          color == "version_x_tier", "Not verified – try colouring by the combined version × tier variable.")
    _clue("c2_km", "A cluster is ~all v2.4 Basic users with very high churn",
          bool(((prof.pct_v24_basic >= .8) & (prof.churn_rate >= .7)).any()),
          "Not verified – no cluster isolates that group yet. Try another k.")

    st.subheader("Issue a verdict")
    if "case2" in st.session_state["solved"]:
        st.success("✅ Case closed: buggy v2.4 on Basic devices caused both slow support and cancellations.")
        st.markdown("**Concept Debrief:** Wait time and churn were correlated only through a *confounder*. "
                    "Within each group the correlation almost vanishes. Clustering can surface hidden segments, "
                    "but only domain thinking tells you which variable is the cause.")
        return
    who = st.selectbox("Root cause of the churn?", SUSPECTS)
    if st.button("⚖️ Submit root cause"):
        if who == SUSPECTS[-1]:
            st.session_state["solved"].add("case2"); st.rerun()
        else:
            _adj(-35); st.error(f"Wrong root cause! '{who}' is a false lead. −35% Credibility")

if __name__ == "__main__":
    _init()
    run_case2()
