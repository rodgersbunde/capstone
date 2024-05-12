import streamlit as st
from surprise import Dataset, Reader, SVD
import pandas as pd
import openai

# Load the dataset
df = pd.read_csv('final_data.csv')

# Set up OpenAI API
openai.api_key = "your_openai_api_key"

# Define Streamlit app content
def streamlit_app():
    # Set page width and background color
    st.set_page_config(page_title="Recipe Search App", layout="wide", page_icon="🥘", initial_sidebar_state="expanded")

    # Create a Streamlit app
    st.title('Recipe Search App')
    st.markdown(
        """
        <style>
        body {
            background-color: #000000; /* Set background color to black */
            color: #FFFFFF; /* Set text color to white */
        }
        .css-1aumxhk {
            background-color: #000000; /* Set sidebar background color to black */
            color: #FFFFFF; /* Set sidebar text color to white */
        }
        .css-1aumxhk a {
            color: #FFFFFF; /* Set sidebar link color to white */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create navigation sidebar with custom colors
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Home', 'About', 'Search'])

    # Display content based on selected page
    if page == 'Home':
        st.write('Welcome to Recipe Search App!')
        st.write('Use the navigation on the left to explore.')
    elif page == 'About':
        st.write('This app helps you search for recipes.')
        st.write('The app returns recipes based on search terms if recipe name is not there it will return recipe name not found')
        st.write('It uses a machine learning model to recommend recipes based on your input.')
    elif page == 'Search':
        st.title('Recipe Search')

        # Add text input for entering search term
        search_term = st.text_input('Enter Search Term:')
        search_button = st.button('Search')

        # Initialize a session state to keep track of displayed recipe names
        if 'displayed_recipe_names' not in st.session_state:
            st.session_state.displayed_recipe_names = set()

        # Filter recipes based on search term and display the results
        if search_term:
            filtered_recipe = df[df['recipe_name'].str.contains(search_term, case=False)]
            if not filtered_recipe.empty:
                for _, row in filtered_recipe.iterrows():
                    if row['recipe_name'] not in st.session_state.displayed_recipe_names:
                        st.session_state.displayed_recipe_names.add(row['recipe_name'])  # Add recipe name to displayed recipe names
                        st.markdown(f"**Recipe Name:** {row['recipe_name']}")
                        st.write(f"**Predicted Rating:** {row['ratings']}")
                        st.markdown(f"**Ingredients:** {row['ingredients']}")
                        st.markdown(f"**Cooking Instructions:** {row['cooking_instructions']}")
                        st.write('---')
            else:
                st.write("Recipe name not found.")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
