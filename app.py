import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.io as pio

# Set page configuration
st.set_page_config(
    page_title="TrueYou Adjective Clustering Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load saved outputs
@st.cache_resource
def load_data():
    # Load the plot from JSON using plotly.io
    with open('cluster_plot.json', 'r') as f:
        fig = pio.from_json(f.read())

    # Load other data
    with open('cluster_labels.json', 'r') as f:
        cluster_labels = json.load(f)

    with open('word_cluster_mapping.json', 'r') as f:
        word_cluster_mapping = json.load(f)

    with open('cluster_words.json', 'r') as f:
        cluster_words = json.load(f)

    # Load plot_df
    plot_df = pd.read_csv('plot_df.csv')

    return fig, cluster_labels, word_cluster_mapping, cluster_words, plot_df

# Load data
fig, cluster_labels, word_cluster_mapping, cluster_words, plot_df = load_data()

# Initialize variables
highlight_cluster_id = None
label = ''
description = ''
words_in_cluster = []

# Sidebar: Toggle between Search and Cluster Selection
st.sidebar.title("Explore Clusters")
option = st.sidebar.radio("Choose an option:", ["Search by Word", "Select Cluster"])

if option == "Search by Word":
    # Sidebar: Search feature
    search_word = st.sidebar.text_input("Enter an adjective to search:")

    if search_word:
        search_word = search_word.lower()
        if search_word in word_cluster_mapping:
            cluster_id = word_cluster_mapping[search_word]
            label = cluster_labels[str(cluster_id)]['label']
            description = cluster_labels[str(cluster_id)]['description']
            words_in_cluster = cluster_words[str(cluster_id)]
            # Exclude the searched word
            other_words = [word for word in words_in_cluster if word.lower() != search_word.lower()]
            st.sidebar.markdown(f"**'{search_word}'** is in **Cluster {cluster_id}: {label}**")
            st.sidebar.markdown(f"**Description:** {description}")
            st.sidebar.markdown(f"**Other words in this cluster:**")
            st.sidebar.markdown('\n'.join(f"- {word}" for word in other_words))
            # Set the cluster ID to highlight
            highlight_cluster_id = cluster_id
        else:
            st.sidebar.markdown(f"**'{search_word}'** not found in any cluster.")
    else:
        st.sidebar.markdown("Enter a word to search for its cluster.")

elif option == "Select Cluster":
    # Prepare cluster information DataFrame
    cluster_info_df = pd.DataFrame.from_dict(cluster_labels, orient='index')
    cluster_info_df.index.name = 'Cluster ID'
    cluster_info_df.reset_index(inplace=True)
    cluster_info_df['Cluster ID'] = cluster_info_df['Cluster ID'].astype(int)

    # Prepare options with cluster IDs and labels
    cluster_info_df['Cluster Option'] = cluster_info_df.apply(
        lambda row: f"Cluster {row['Cluster ID']}: {row['label']}", axis=1
    )
    cluster_options = cluster_info_df['Cluster Option'].tolist()

    # Create a mapping from display text to cluster ID
    cluster_mapping = {option: row['Cluster ID'] for option, row in zip(cluster_options, cluster_info_df.to_dict('records'))}

    # Sidebar: Selectbox with formatted cluster options
    selected_option = st.sidebar.selectbox("Select a Cluster to view details:", cluster_options)

    # Get the cluster ID from the selected option
    selected_cluster_id = cluster_mapping[selected_option]

    # Set the cluster ID to highlight
    highlight_cluster_id = selected_cluster_id

    # Get cluster details
    label = cluster_labels[str(selected_cluster_id)]['label']
    description = cluster_labels[str(selected_cluster_id)]['description']
    words_in_cluster = cluster_words[str(selected_cluster_id)]

    # Display cluster details in the sidebar
    st.sidebar.markdown(f"### Cluster {selected_cluster_id}: {label}")
    st.sidebar.markdown(f"**Description:** {description}")
    st.sidebar.markdown("**Words in this cluster:**")
    st.sidebar.markdown('\n'.join(f"- {word}" for word in words_in_cluster))

# If no cluster is selected, set highlight_cluster_id to -1
if highlight_cluster_id is None:
    highlight_cluster_id = -1  # No cluster to highlight

# Now, we'll create containers to control the layout
# Create a container for the plot
plot_container = st.container()

# Update the figure to highlight the selected cluster
def update_figure_with_highlight(fig, plot_df, cluster_id):
    # Create a copy of the figure
    fig_updated = go.Figure(fig)

    # Create color and size arrays
    colors = ['blue'] * len(plot_df)
    sizes = [8] * len(plot_df)

    # Find the index of the selected cluster
    selected_index = plot_df.index[plot_df['cluster_id'] == cluster_id].tolist()
    if selected_index:
        idx = selected_index[0]
        colors[idx] = 'red'
        sizes[idx] = 15

    # Update the marker colors and sizes
    fig_updated.data[0].marker.color = colors
    fig_updated.data[0].marker.size = sizes

    return fig_updated

# Update the figure
fig_updated = update_figure_with_highlight(fig, plot_df, highlight_cluster_id)

# Display the plot in the plot_container
with plot_container:
    st.title("TrueYou Adjective Clustering Analysis")

    st.plotly_chart(fig_updated, use_container_width=True)
    st.markdown("[See here for interpretation of X and Y axes (first two principal components)](https://docs.google.com/document/d/1yYEmSKJsj-I8pu1CAxYqRpVFbmANXTR6364mMuVNWek/edit?usp=sharing)")
    # Place the expander with the cluster labels and descriptions between the title and the plot
    with st.expander("Show Cluster Labels and Descriptions"):
        cluster_info_df = pd.DataFrame.from_dict(cluster_labels, orient='index')
        cluster_info_df.index.name = 'Cluster ID'
        cluster_info_df.reset_index(inplace=True)
        cluster_info_df['Cluster ID'] = cluster_info_df['Cluster ID'].astype(int)
        st.dataframe(cluster_info_df[['Cluster ID', 'label', 'description']], use_container_width=True)
