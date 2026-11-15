import streamlit as st
from case1 import run_case1
from case2 import run_case2
from case3 import run_case3

st.set_page_config(page_title="Data Files – The Unknown", page_icon="🕵️", layout="wide")
st.session_state.setdefault("credibility", 100)
st.session_state.setdefault("solved", set())

CASES = {"Case 001: The Vanishing Revenue": ("case1", run_case1),
         "Case 002: The Silent Customers": ("case2", run_case2),
         "Case 003: The Power Surge": ("case3", run_case3)}

def restart():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

st.sidebar.title("🕵️ Data Files")
meter = st.sidebar.empty()
def draw_meter():
    c = st.session_state["credibility"]
    meter.metric("Credibility Meter", f"{c}%")
    meter.progress(c / 100)

if st.session_state["credibility"] < 40:
    draw_meter()
    st.title("🚨 Wrongful Accusation / Grid Failure")
    st.error("Your credibility fell below 40%. The Bureau has pulled you from the investigation.")
    st.markdown("""### Post-mortem: what went wrong
- **Duplicates + biased sampling (Case 001):** duplicate refunds and a single 7-day window made one region look guilty. A sample must represent the whole population.
- **Correlation ≠ causation (Case 002):** wait time and churn moved together only because a hidden variable (buggy app version on Basic devices) drove both.
- **Bias–variance (Case 003):** too-simple models miss the surge (blackout); too-flexible models memorise noise and fail on new data (false shutdown).
""")
    st.button("🔁 Restart investigation", on_click=restart)
    st.stop()

choice = st.sidebar.radio("Select case file", list(CASES.keys()))
with st.sidebar.expander("📖 How to Play & Rules", expanded=False):
    st.markdown("""- Explore charts, filters and models **freely – it costs nothing**.
- Saving a **verified data clue**: **+5%** credibility.
- Accusing the wrong suspect or deploying a bad model: **−35%**.
- **75–100%** → Master Detective · **40–74%** → Sloppy Investigation · **<40%** → Wrongful Accusation / Grid Failure.
- Each case opens with Learning Objectives and closes with a Concept Debrief.""")
st.sidebar.write("Solved: " + (", ".join(sorted(s for s in st.session_state["solved"] if not s.startswith("clue:"))) or "none yet"))
st.sidebar.button("🔁 Restart game", on_click=restart)

CASES[choice][1]()
draw_meter()

if st.session_state["credibility"] < 40:
    st.rerun()
if {"case1", "case2", "case3"} <= st.session_state["solved"]:
    st.divider()
    c = st.session_state["credibility"]
    if c >= 75:
        st.balloons(); st.success(f"🏆 MASTER DETECTIVE – all cases closed with {c}% credibility. Clean data, sound models, correct culprits.")
    else:
        st.warning(f"📎 SLOPPY INVESTIGATION – all cases closed, but only {c}% credibility. Review the debriefs and try again.")
