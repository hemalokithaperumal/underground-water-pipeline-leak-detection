
        st.divider()

        st.subheader("Project Team")

        team_df = pd.DataFrame({
            "Role": [
                "Project Student",
                "Project Student"
            ],
            "Name": [
                "Hema Lokitha P",
                "Gayathiri Roshini M"
            ]
        })

        st.dataframe(
            team_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Project Guide")

        st.markdown("""
        **Mr. K. Premnath**

        Department of Computer Science and Engineering
        """)

        st.divider()

        st.subheader("Technology Stack")

        technology_df = pd.DataFrame({
            "Technology": [
                "Python",
                "NumPy",
                "Pandas",
                "GraphSAGE-v2",
                "Streamlit",
                "Plotly",
                "GitHub",
                "Streamlit Community Cloud"
            ],
            "Purpose": [
                "Programming language",
                "Numerical computation",
                "Dataset handling",
                "Graph-based learning model",
                "Web application framework",
                "Interactive charts",
                "Source code management",
                "Cloud deployment"
            ]
        })

        st.dataframe(
            technology_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Project Scope")

        st.markdown("""
        This website demonstrates an experimental
        multi-sensor pipeline classification model.

        The current deployment uses prepared experimental
        samples and an exported trained model.

        Live sensor acquisition, physical sensor installation,
        real-time monitoring and edge deployment are outside
        the scope of this demonstration.
        """)

    # =================================================
    # 9. COMMON FOOTER
    # =================================================

    st.divider()

    with st.expander("ℹ️ Project Scope and Limitations"):

        st.markdown("""
        - This demonstration uses prepared experimental dataset samples.
        - The website does not currently process live sensor signals.
        - It does not provide real-time pipeline monitoring.
        - Physical sensor installation, live streaming and
          edge deployment are outside the scope of this demonstration.
        - Model performance on experimental samples may differ
          from performance in real underground pipeline environments.
        """)

    st.caption(
        "PIPELINE AI | Academic Final Year Project | GraphSAGE-v2"
    )


except Exception as e:

    st.error(
        f"An error occurred while loading or running the application: {e}"
    )

    st.info(
        "Please verify that demo_model..json exists in your "
        "GitHub repository and that the filename matches "
        "the filename used in load_model()."
    )
