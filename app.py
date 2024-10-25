import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Adjective Clusters",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load saved outputs
@st.cache_resource
def load_data():
    # Load the plot from JSON
    with open('cluster_plot.json', 'r') as f:
        fig = go.Figure.from_json(f.read())

    # Load cluster labels and descriptions
    with open('cluster_labels.json', 'r') as f:
        cluster_labels = json.load(f)

    # Load word-to-cluster mapping
    with open('word_cluster_mapping.json', 'r') as f:
        word_cluster_mapping = json.load(f)

    # Load cluster descriptions
    with open('cluster_descriptions.json', 'r') as f:
        cluster_descriptions = json.load(f)

    # Load cluster words
    with open('cluster_words.json', 'r') as f:
        cluster_words = json.load(f)

    return fig, cluster_labels, word_cluster_mapping, cluster_descriptions, cluster_words

# Load data
fig, cluster_labels, word_cluster_mapping, cluster_descriptions, cluster_words = load_data()

# Display the title
st.title("Adjective Clustering Analysis")

# Display the plot
st.subheader("Interactive Cluster Plot")
st.plotly_chart(fig, use_container_width=True)

# Display LLM outputs (cluster labels and descriptions)
st.subheader("Cluster Labels and Descriptions")
cluster_info_df = pd.DataFrame.from_dict(cluster_labels, orient='index')
cluster_info_df.index.name = 'Cluster ID'
cluster_info_df.reset_index(inplace=True)
st.dataframe(cluster_info_df)

# Search feature
st.sidebar.title("Search Word in Clusters")
search_word = st.sidebar.text_input("Enter an adjective to search:")

if search_word:
    search_word = search_word.lower()
    if search_word in word_cluster_mapping:
        cluster_id = word_cluster_mapping[search_word]
        label = cluster_labels[str(cluster_id)]['label']
        description = cluster_labels[str(cluster_id)]['description']
        st.sidebar.markdown(f"**'{search_word}'** is in **Cluster {cluster_id}: {label}**")
        st.sidebar.markdown(f"**Description:** {description}")

        # Show other words in the same cluster
        st.sidebar.markdown(f"**Other words in this cluster:**")
        # For this, we need to load a mapping from clusters to words
        @st.cache_resource
        def load_cluster_words():
            # Since the full DataFrame is large, we'll load the necessary data from the outputs
            # We need to have saved this mapping during the analysis
            with open('cluster_words.json', 'r') as f:
                cluster_words = json.load(f)
            return cluster_words

        cluster_words = load_cluster_words()
        words_in_cluster = cluster_words[str(cluster_id)]
        st.sidebar.write(', '.join(words_in_cluster))
    else:
        st.sidebar.markdown(f"**'{search_word}'** not found in any cluster.")
else:
    st.sidebar.markdown("Enter a word to search for its cluster.")

# Optionally, display the description of a selected cluster
st.subheader("Explore Clusters")
selected_cluster_id = st.selectbox("Select a Cluster ID to view details:", cluster_info_df['Cluster ID'])

if selected_cluster_id is not None:
    label = cluster_labels[str(selected_cluster_id)]['label']
    description = cluster_labels[str(selected_cluster_id)]['description']
    st.markdown(f"### Cluster {selected_cluster_id}: {label}")
    st.markdown(f"**Description:** {description}")

    # Show words in this cluster
    # Load cluster words if not already loaded
    @st.cache_resource
    def load_cluster_words():
        with open('cluster_words.json', 'r') as f:
            cluster_words = json.load(f)
        return cluster_words

    cluster_words = load_cluster_words()
    words_in_cluster = cluster_words[str(selected_cluster_id)]
    st.markdown("**Words in this cluster:**")
    st.write(', '.join(words_in_cluster))
