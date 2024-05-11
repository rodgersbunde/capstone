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
    st.set_page_config(page_title="Recipe Recommendation App", layout="wide", page_icon="🥘", initial_sidebar_state="expanded")

    # Create a Streamlit app
    st.title('Recipe Recommendation App')
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
    page = st.sidebar.radio('Go to', ['Home', 'About', 'Results'])

    # Display content based on selected page
    if page == 'Home':
        st.write('Welcome to Recipe Search App!')
        st.write('Use the navigation on the left to explore.')
    elif page == 'About':
        st.write('This app helps you search for recipes based on the highly rated recipes.')
        st.write('The app returns the top rated recipes')
        st.write('It uses a machine learning model to recommend recipes based on your input.')
    elif page == 'Results':
        st.title('Recipe Search Results')

        # Load collaborative filtering model
        reader = Reader(rating_scale=(0, 100))
        data = Dataset.load_from_df(df[['user_id', 'recipe_code', 'ratings']], reader)
        trainset = data.build_full_trainset()
        algo = SVD()
        algo.fit(trainset)

        # Function to get recommendations for a user
        def get_recommendations(user_id, df, recipe_name):
            user_recipe = df[df['user_id'] == user_id]['recipe_code'].unique()
            recommended_recipe = []
            for recipe_code in df['recipe_code'].unique():
                if recipe_code not in user_recipe:
                    predicted_ratings = algo.predict(user_id, recipe_code).est
                    recommended_recipe.append((recipe_code, predicted_ratings))
            recommended_recipe.sort(key=lambda x: x[1], reverse=True)
            return recommended_recipe

        # Add text inputs for entering user ID and recipe name
        user_id = st.text_input('Enter User ID:')
        recipe_name = st.text_input('Enter Recipe Name:')
        search_term = st.text_input('Enter Search Term:')
        search_button = st.button('Search')
        # Get recommendations for the user
        if user_id and recipe_name:
            recommended_recipe = get_recommendations(int(user_id), df, recipe_name)
            if not recommended_recipe:
                # Use OpenAI to generate response for unknown recipe names
                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=f"Provide additional context or details for the recipe: {recipe_name}",
                    max_tokens=50,
                    n=1,
                    stop=None,
                    temperature=0.5,
                )
                st.markdown(f"**Recipe Name:** {recipe_name}")
                st.write("Predicted Rating: Not available")
                st.markdown("**Ingredients:** Not available")
                st.markdown("**Cooking Instructions:** Not available")
            else:
                for recipe_code, predicted_rating in recommended_recipe[:20]:
                    recipe_name = df[df['recipe_code'] == recipe_code]['recipe_name'].iloc[0]
                    ingredients = df[df['recipe_code'] == recipe_code]['ingredients'].iloc[0]
                    # Check if cooking instructions exist before accessing them
                    if 'cooking_instructions' in df.columns and not df[df['recipe_code'] == recipe_code]['cooking_instructions'].empty:
                        cooking_instructions = df[df['recipe_code'] == recipe_code]['cooking_instructions'].iloc[0]
                        st.markdown(f"<span style='color: #FFD700'><strong>Recipe Name:</strong></span> {recipe_name}", unsafe_allow_html=True)
                        st.write(f"<span style='color: #00FF00'><strong>Predicted Rating:</strong></span> {predicted_rating}", unsafe_allow_html=True)
                        st.markdown(f"<span style='color: #FFA07A'><strong>Ingredients:</strong></span> {ingredients}", unsafe_allow_html=True)
                        st.markdown(f"<span style='color: #87CEEB'><strong>Cooking Instructions:</strong></span> {cooking_instructions}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color: #FFD700'><strong>Recipe Name:</strong></span> {recipe_name}", unsafe_allow_html=True)
                        st.write(f"<span style='color: #00FF00'><strong>Predicted Rating:</strong></span> {predicted_rating}", unsafe_allow_html=True)
                        st.markdown(f"<span style='color: #FFA07A'><strong>Ingredients:</strong></span> {ingredients}", unsafe_allow_html=True)
                        st.markdown("<span style='color: #87CEEB'><strong>Cooking Instructions:</strong></span> Not available", unsafe_allow_html=True)

        # Filter recipes based on search term and display only the first result
        #search_term = st.text_input('Enter Search Term:')
        if search_term:
            filtered_recipe = df[df['recipe_name'].str.contains(search_term, case=False)].head(1)
            if not filtered_recipe.empty:
                st.markdown(f"**Search Result for {search_term}**")
                for index, row in filtered_recipe.iterrows():
                    st.markdown(f"<span style='color: #FFD700'><strong>Recipe Name:</strong></span> {row['recipe_name']}", unsafe_allow_html=True)
                    st.write(f"<span style='color: #00FF00'><strong>Predicted Rating:</strong></span> {row['ratings']}", unsafe_allow_html=True)
                    st.markdown(f"<span style='color: #FFA07A'><strong>Ingredients:</strong></span> {row['ingredients']}", unsafe_allow_html=True)
                    st.markdown(f"<span style='color: #87CEEB'><strong>Cooking Instructions:</strong></span> {row['cooking_instructions']}", unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()