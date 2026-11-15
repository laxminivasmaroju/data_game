import numpy as np, pandas as pd, plotly.express as px, streamlit as st

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
    rng = np.random.default_rng(7)
    n = 450
    df = pd.DataFrame({"order_id": np.arange(1000, 1000 + n), "day": rng.integers(1, 29, n),
                       "region": rng.choice(["North", "South", "East", "West"], n),
                       "amount": rng.normal(120, 25, n).round(2).clip(20, None)})
    df["status"] = np.where(rng.random(n) < 0.06, "Refund", "Sale")
    spike = (df.region == "North") & df.day.between(8, 14) & (rng.random(n) < 0.7)
    df.loc[spike, "status"] = "Refund"
    outage = (df.region == "South") & (df.day >= 15) & (rng.random(n) < 0.65)
    df["payment_status"] = np.where(outage, "GATEWAY_TIMEOUT", "OK")
    df.loc[outage | (rng.random(n) < 0.02), "amount"] = np.nan
    dups = df[spike].head(50)
    dups = pd.concat([dups, df.drop(dups.index).sample(50 - len(dups), random_state=1)])
    out = pd.concat([df, dups]).sample(frac=1, random_state=2).reset_index(drop=True)
    out["week"] = (out.day - 1) // 7 + 1
    return out

SUSPECTS = ["North Region Manager", "East Marketing Lead", "West Courier Partner",
            "South warehouse payment-gateway outage"]

def run_case1():
    _init()
    st.title("Case 001: The Vanishing Revenue")
    st.info("**Learning Objectives:** clean messy data (duplicates, missing values), recognise sampling bias, "
            "and use EDA to find the real cause of a revenue drop.")
    st.write("Company revenue has dropped. Rumour says the **North Region Manager** is fudging refunds. "
             "Inspect the 500-row ledger before accusing anyone.")
    df = make_data()
    with st.expander("🔎 Raw ledger (first 25 rows)"):
        st.dataframe(df.head(25), use_container_width=True)
    a, b, c = st.columns(3)
    sample = a.radio("Sampling frame", ["Biased 7-day sample (days 8–14)", "Full stratified sample (all 28 days)"])
    dedupe = b.checkbox("Remove duplicate rows", False)
    miss = c.radio("Missing amounts", ["Keep as blank (counts as $0)", "Drop rows", "Fill with median"])

    s = df[df.day.between(8, 14)] if sample.startswith("Biased") else df
    if dedupe:
        s = s.drop_duplicates()
    d = s
    if miss.startswith("Drop"):
        d = d.dropna(subset=["amount"])
    elif miss.startswith("Fill"):
        d = d.assign(amount=d.amount.fillna(d.amount.median()))
    d = d.assign(net=d.amount.fillna(0) * np.where(d.status == "Refund", -1, 1))

    m1, m2, m3 = st.columns(3)
    m1.metric("Rows in view", len(s)); m2.metric("Exact duplicates", int(s.duplicated().sum()))
    m3.metric("Missing amounts", int(s.amount.isna().sum()))
    g1, g2 = st.columns(2)
    rev = d.groupby("region", as_index=False).net.sum()
    g1.plotly_chart(px.bar(rev, x="region", y="net", color="region", title="Net revenue by region"), use_container_width=True)
    mr = s.assign(missing=s.amount.isna()).groupby(["week", "region"], as_index=False).missing.mean()
    g2.plotly_chart(px.bar(mr, x="week", y="missing", color="region", barmode="group",
                           title="Share of missing payments by week & region"), use_container_width=True)

    st.subheader("Evidence board")
    _clue("c1_dups", "North's refunds were duplicated entries", dedupe,
          "Not verified yet – remove the duplicate rows and compare North's net revenue.")
    _clue("c1_south", "South payments fail in weeks 3–4 (visible only in the full sample)",
          sample.startswith("Full"), "Not verified – a 7-day window can't show a spike in other weeks. Widen the sample.")

    st.subheader("Issue a verdict")
    if "case1" in st.session_state["solved"]:
        st.success("✅ Case closed: the South warehouse payment gateway timed out, leaving payments blank.")
        st.markdown("**Concept Debrief:** Duplicates inflate counts, a biased sample frame can frame the innocent, "
                    "and missing data is rarely random – the *pattern* of missingness (South, weeks 3–4) was the clue. "
                    "Filling blanks with a median would have hidden it.")
        return
    who = st.selectbox("Who/what caused the revenue drop?", SUSPECTS)
    if st.button("⚖️ Issue arrest warrant"):
        if who == SUSPECTS[-1]:
            st.session_state["solved"].add("case1"); st.rerun()
        else:
            _adj(-35); st.error(f"Wrongful accusation! The evidence does not support '{who}'. −35% Credibility")

if __name__ == "__main__":
    _init()
    run_case1()
