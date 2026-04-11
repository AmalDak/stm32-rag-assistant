# frontend/app.py
# Streamlit UI — talks to the FastAPI backend.

import streamlit as st
import requests

API_URL = "http://backend:8000/query"  # Docker service name

st.set_page_config(
    page_title="STM32 Consultant",
    page_icon="🔧",
    layout="centered"
)

st.title("STM32 Pre-Sales Consultant")
st.caption(
    "Describe your embedded system requirements and get a "
    "chip recommendation grounded in official documentation."
)

# --- Query input --------------------------------------------
query = st.text_area(
    "Your requirement",
    placeholder="e.g. I need a low power MCU with BLE for a wearable device",
    height=100
)

if st.button("Get recommendation", type="primary"):
    if not query.strip():
        st.warning("Please enter a requirement first.")
    else:
        with st.spinner("Consulting documentation..."):
            try:
                resp = requests.post(
                    API_URL,
                    json={"query": query},
                    timeout=180
                )
                resp.raise_for_status()
                data = resp.json()

            except requests.RequestException as e:
                st.error(f"Could not reach the backend: {e}")
                st.stop()

        # --- Answer -----------------------------------------
        st.subheader("Recommendation")
        st.markdown(data["answer"])

        # --- Extracted requirements -------------------------
        with st.expander("Extracted requirements"):
            st.json(data["requirements"])

        # --- Candidate chips --------------------------------
        with st.expander("Candidate chips considered"):
            for chip in data["candidates"]:
                st.markdown(
                    f"**{chip['chip']}** — {chip['family']} | "
                    f"{chip['max_freq_mhz']}MHz | "
                    f"{chip['flash_kb']}KB flash | "
                    f"Board: {chip['dev_board']}"
                )

        # --- Sources ----------------------------------------
        with st.expander("Documentation sources used"):
            for i, src in enumerate(data["sources"], start=1):
                st.markdown(
                    f"[{i}] `{src['file']}` — "
                    f"page {src['page']} "
                    f"(score: {src['score']})"
                )