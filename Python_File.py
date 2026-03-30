

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import questionary
import sys

class MovieRecommender:
    def __init__(self, file_path):
        """
        Sets up the movie database and builds the AI models right when the program starts.
        """
        try:
            self.df = pd.read_csv(file_path)
            
            # Clean up column names just in case there are weird spaces in the CSV
            self.df.columns = [col.strip().title() for col in self.df.columns]
            if 'Movie name' in self.df.columns:
                self.df.rename(columns={'Movie name': 'Movie Name'}, inplace=True)
                
            self._prepare_data()
            self._build_ai_model()
            
        except FileNotFoundError:
            print(f"\n[!] Oops! Couldn't find '{file_path}'. Make sure it's in the same folder.")
            sys.exit(1)
        except Exception as e:
            print(f"\n[!] Something went wrong loading the data: {e}")
            sys.exit(1)

    def _prepare_data(self):
        """
        Fills in missing data and combines all text into one big 'context' string for the AI to read.
        """
        features = ['Genre', 'Director', 'Cast', 'Language', 'Year']
        
        # Replace empty spreadsheet cells with empty strings so the math doesn't break
        for feature in features:
            if feature in self.df.columns:
                self.df[feature] = self.df[feature].fillna('')
            else:
                self.df[feature] = ''

        # Combine the text features into one column
        def combine_features(row):
            return f"{row['Genre']} {row['Director']} {row['Cast']}".lower()

        self.df['ai_context'] = self.df.apply(combine_features, axis=1)

    def _build_ai_model(self):
        """
        Uses TF-IDF to figure out which keywords are actually unique and important.
        """
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df['ai_context'])
        
        # Calculate how mathematically similar every movie is to every other movie
        self.similarity_scores = cosine_similarity(tfidf_matrix)

    def find_similar_movies(self, user_movie, top_n=5):
        """
        Finds movies based on AI similarity, with built-in typo forgiveness.
        """
        all_movie_titles = self.df['Movie Name'].tolist()
        
        # AI Typo Correction: Find the closest real movie title (requires 60% match)
        close_matches = difflib.get_close_matches(user_movie, all_movie_titles, n=1, cutoff=0.6)
        
        if not close_matches:
            print(f"\n[?] We couldn't find anything matching '{user_movie}'.")
            return

        # Grab the exact title the AI decided on
        best_match = close_matches[0]
        if best_match.lower() != user_movie.lower():
            print(f"\n(AI Auto-Corrected '{user_movie}' to '{best_match}')")
            
        # Look up the movie's ID number in our dataframe
        movie_idx = self.df[self.df['Movie Name'] == best_match].index[0]
        
        # Get the similarity scores for this specific movie and sort them highest to lowest
        scores = list(enumerate(self.similarity_scores[movie_idx]))
        sorted_scores = sorted(scores, reverse=True, key=lambda x: x[1])
        
        print(f"\n🎬 If you liked '{best_match}', you should check out:")
        print("-" * 45)
        
        results_shown = 0
        for index, score in sorted_scores:
            if index == movie_idx:
                continue # Skip the movie the user actually searched for
                
            title = self.df.iloc[index]['Movie Name']
            genre = self.df.iloc[index]['Genre']
            
            print(f"{results_shown + 1}. {title} ({genre})")
            
            results_shown += 1
            if results_shown >= top_n:
                break
        print("-" * 45)

    def browse_by_category(self, category, search_term, top_n=10):
        """
        Filters the database based on a text search within a specific column.
        """
        category = category.title()
        
        if category not in self.df.columns:
            print(f"\n[!] The category '{category}' doesn't exist in the data.")
            return

        # Search the column for partial, case-insensitive matches
        matches = self.df[self.df[category].astype(str).str.contains(search_term, case=False, na=False)]
        
        if matches.empty:
            print(f"\n[?] No movies found matching '{search_term}' in the {category} category.")
            return
            
        print(f"\n📂 Top results for '{search_term}' in {category}:")
        print("-" * 45)
        
        for count, (_, row) in enumerate(matches.head(top_n).iterrows()):
            title = row['Movie Name']
            year = row.get('Year', 'N/A')
            director = row.get('Director', 'N/A')
            print(f"{count + 1}. {title} (Year: {year} | Dir: {director})")
            
        print("-" * 45)

# ==========================================
# MAIN INTERACTIVE APP
# ==========================================
if __name__ == "__main__":
    recommender = MovieRecommender('movies.csv')
    
    while True:
        # Use Questionary to create a beautiful interactive terminal menu
        action = questionary.select(
            "🍿 Welcome to the AI Movie Recommender! What would you like to do?",
            choices=[
                "🔍 Find movies similar to a favorite",
                "📂 Browse movies by a specific category",
                "❌ Exit Program"
            ]
        ).ask()
        
        # Handle the user's menu choice
        if action == "🔍 Find movies similar to a favorite":
            movie_input = questionary.text("Enter a movie name (typos are okay!):").ask()
            if movie_input:
                recommender.find_similar_movies(movie_input)
                
        elif action == "📂 Browse movies by a specific category":
            category = questionary.select(
                "Which category do you want to search?",
                choices=["Genre", "Director", "Language", "Year", "Cast"]
            ).ask()
            
            if category:
                search_term = questionary.text(f"Enter the {category} you are looking for:").ask()
                if search_term:
                    recommender.browse_by_category(category, search_term)
                
        elif action == "❌ Exit Program" or action is None:
            print("\nThanks for using the AI Movie Recommender. Goodbye! 👋\n")
            break
            
        # Add a pause before clearing the screen/looping back
        input("\nPress Enter to return to the main menu...")

