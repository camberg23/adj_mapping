import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.io as pio  # Import plotly.io

# Set page configuration
st.set_page_config(
    page_title="Adjective Clustering Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load saved outputs
@st.cache_resource
def load_data():
    # Load the plot from JSON using plotly.io
    with open('cluster_plot.json', 'r') as f:
        fig = pio.from_json(f.read())

    # Load other data as before
    with open('cluster_labels.json', 'r') as f:
        cluster_labels = json.load(f)

    with open('word_cluster_mapping.json', 'r') as f:
        word_cluster_mapping = json.load(f)

    with open('cluster_descriptions.json', 'r') as f:
        cluster_descriptions = json.load(f)

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
st.dataframe(cluster_info_df, use_container_width=True)  # Updated to use full width

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
        words_in_cluster = cluster_words[str(cluster_id)]
        # Exclude the searched word
        other_words = [word for word in words_in_cluster if word.lower() != search_word.lower()]
        st.sidebar.write(', '.join(other_words))
    else:
        st.sidebar.markdown(f"**'{search_word}'** not found in any cluster.")
else:
    st.sidebar.markdown("Enter a word to search for its cluster.")

# Optionally, display the description of a selected cluster
st.subheader("Explore Clusters")

# Prepare options with cluster IDs and labels
cluster_info_df['Cluster Option'] = cluster_info_df.apply(
    lambda row: f"Cluster {row['Cluster ID']}: {row['label']}", axis=1
)
cluster_options = cluster_info_df['Cluster Option'].tolist()

# Create a mapping from display text to cluster ID
cluster_mapping = {option: row['Cluster ID'] for option, row in zip(cluster_options, cluster_info_df.to_dict('records'))}

# Selectbox with formatted cluster options
selected_option = st.selectbox("Select a Cluster to view details:", cluster_options)

# Get the cluster ID from the selected option
selected_cluster_id = cluster_mapping[selected_option]

if selected_cluster_id is not None:
    label = cluster_labels[str(selected_cluster_id)]['label']
    description = cluster_labels[str(selected_cluster_id)]['description']
    st.markdown(f"### Cluster {selected_cluster_id}: {label}")
    st.markdown(f"**Description:** {description}")

    # Show words in this cluster
    words_in_cluster = cluster_words[str(selected_cluster_id)]
    st.markdown("**Words in this cluster:**")
    st.markdown('\n'.join(f"- {word}" for word in words_in_cluster))  # Display words as a bulleted list
