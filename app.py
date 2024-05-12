import streamlit as st
import pandas as pd

# Load the dataset
df = pd.read_csv('final_data.csv')

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
        st.write('This app helps you search for recipes .')
        st.write('The app returns recipes based on search terms if recipe name is not it will return recipe name not found.')
        st.write('It uses a machine learning model to recommend recipes based on your input.')
    elif page == 'Search':
        st.title('Recipe Search')

        # Add text input for entering search term
        search_term = st.text_input('Enter Search Term:')
        search_button = st.button('Search')

        # Filter recipes based on search term and display the results
        if search_term:
            filtered_recipe = df[df['recipe_name'].str.contains(search_term, case=False)]
            if not filtered_recipe.empty:
                for _, row in filtered_recipe.iterrows():
                    st.markdown(f"<span style='color: #FFD700'><strong>Recipe Name:</strong></span> {row['recipe_name']}", unsafe_allow_html=True)
                    st.write(f"<span style='color: #00FF00'><strong>Predicted Rating:</strong></span> {row['ratings']}", unsafe_allow_html=True)
                    st.markdown(f"<span style='color: #FFA07A'><strong>Ingredients:</strong></span> {row['ingredients']}", unsafe_allow_html=True)
                    st.markdown(f"<span style='color: #87CEEB'><strong>Cooking Instructions:</strong></span> {row['cooking_instructions']}", unsafe_allow_html=True)
                    st.write('---')
            else:
                st.write("Recipe name not found.")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
