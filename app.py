import streamlit as st
from surprise import Dataset, Reader, SVD
import pandas as pd
from fuzzywuzzy import process

# Load the dataset
df = pd.read_csv('final_data.csv')

# Define Streamlit app content
def streamlit_app():
    # Set page width and background color
    st.set_page_config(page_title="Recipe Recommendation App", layout="wide", page_icon="🥘", initial_sidebar_state="expanded")

    # Create a Streamlit app
    st.title('Recipe Recommendation App')
    st.markdown(
        """
        <style>
        .css-1aumxhk {
            background-color: #f0f2f6;
        }
        .css-1aumxhk a {
            color: #333;
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
        st.write('This app helps users to search for recipes by key in `Recipe Name`.')
        st.write('Key in `User ID` which is any integer.')
        st.write('It uses a machine learning model to recommend top 10 recipes based on high ratings.')
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
            # Fuzzy matching to find similar recipe names
            similar_recipe_name, _ = process.extractOne(recipe_name, df['recipe_name'])
            # Filter dataframe based on similar recipe name
            filtered_df = df[df['recipe_name'] == similar_recipe_name]
            user_recipe = filtered_df[filtered_df['user_id'] == user_id]['recipe_code'].unique()
            recommended_recipe = []
            for recipe_code in filtered_df['recipe_code'].unique():
                if recipe_code not in user_recipe:
                    predicted_ratings = algo.predict(user_id, recipe_code).est
                    recommended_recipe.append((recipe_code, predicted_ratings))
            recommended_recipe.sort(key=lambda x: x[1], reverse=True)
            return recommended_recipe

        # Add text inputs for entering user ID and recipe name
        user_id = st.text_input('Enter User ID:')
        recipe_name = st.text_input('Enter Recipe Name:')
        search_button = st.button('Search')
        # Get recommendations for the user
        if user_id and recipe_name:
            recommended_recipe = get_recommendations(int(user_id), df, recipe_name)
            for recipe_code, predicted_rating in recommended_recipe[:10]:
                recipe_name = df[df['recipe_code'] == recipe_code]['recipe_name'].iloc[0]
                ingredients = df[df['recipe_code'] == recipe_code]['ingredients'].iloc[0]
                # Check if cooking instructions exist before accessing them
                if 'cooking_instructions' in df.columns and not df[df['recipe_code'] == recipe_code]['cooking_instructions'].empty:
                    cooking_instructions = df[df['recipe_code'] == recipe_code]['cooking_instructions'].iloc[0]
                    st.write(f"Recipe Name: {recipe_name}")
                    st.write(f"Predicted Rating: {predicted_rating}")
                    st.write(f"Ingredients: {ingredients}")
                    st.write(f"Cooking Instructions: {cooking_instructions}")
                else:
                    st.write(f"Recipe Name: {recipe_name}")
                    st.write(f"Predicted Rating: {predicted_rating}")
                    st.write(f"Ingredients: {ingredients}")
                    st.write("Cooking Instructions not available for this recipe.")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
