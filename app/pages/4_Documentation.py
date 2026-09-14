import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Documentation", page_icon="📚")

st.title("📚 Documentation")

# __file__       = app/pages/4_Documentation.py
# .parents[0]    = app/pages/
# .parents[1]    = app/
# .parents[2]    = the project root directory!
root_dir = Path(__file__).resolve().parents[2]

tab1, tab2 = st.tabs(["README", "Architecture"])

with tab1:
    readme_path = root_dir / "README.md"
    if readme_path.exists():
        # Added encoding="utf-8" to prevent emoji decoding errors on Windows
        st.markdown(readme_path.read_text(encoding="utf-8"))
    else:
        st.error(f"README.md not found. Looked in: {readme_path}")

with tab2:
    arch_path = root_dir / "docs" / "architecture.md"
    if arch_path.exists():
        st.markdown(arch_path.read_text(encoding="utf-8"))
    else:
        st.error(f"Architecture docs not found. Looked in: {arch_path}")