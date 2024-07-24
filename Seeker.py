import sys
import requests
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QTextEdit, QLabel, QMessageBox
from PyQt5.QtCore import Qt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

class SearchClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.vectorizer = None
        self.clf = None
        self.search_terms = [
            "Nike", "Adidas", "Manchester United",  # Sports
            "Apple", "Microsoft", "Python programming",  # Technology
            "India", "France", "United States",  # Countries
            "Leonardo da Vinci", "Vincent van Gogh", "Louvre Museum",  # Art
            "William Shakespeare", "To Kill a Mockingbird", "George Orwell",  # Literature
            "Albert Einstein", "Theory of Relativity", "Quantum Physics",  # Science
            "The Beatles", "Mozart", "Jazz music",  # Music
            "Climate change", "Renewable energy", "Biodiversity",  # Environment
            "World War II", "Industrial Revolution", "Ancient Egypt",  # History
            "Bitcoin", "Stock market", "Entrepreneurship",  # Finance/Business
            "Taj Mahal", "Great Wall of China", "Eiffel Tower",  # Landmarks
            "Vegetarianism", "Mediterranean diet", "Fast food",  # Food
            "Yoga", "Meditation", "Cardiovascular exercise",  # Health/Fitness
            "Hollywood", "Bollywood", "Film directors",  # Entertainment
            "Democracy", "United Nations", "Human rights"  # Politics/Society
        ]
        self.categories = [
            "Sports", "Sports", "Sports",
            "Technology", "Technology", "Technology",
            "Geography", "Geography", "Geography",
            "Art", "Art", "Art",
            "Literature", "Literature", "Literature",
            "Science", "Science", "Science",
            "Music", "Music", "Music",
            "Environment", "Environment", "Environment",
            "History", "History", "History",
            "Finance/Business", "Finance/Business", "Finance/Business",
            "Landmarks", "Landmarks", "Landmarks",
            "Food", "Food", "Food",
            "Health/Fitness", "Health/Fitness", "Health/Fitness",
            "Entertainment", "Entertainment", "Entertainment",
            "Politics/Society", "Politics/Society", "Politics/Society"
        ]
        
    def initUI(self):
        self.setWindowTitle('Search Classifier')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        # Training section
        train_layout = QHBoxLayout()
        self.term_input = QLineEdit()
        self.term_input.setPlaceholderText("Enter search term")
        self.category_input = QLineEdit()
        self.category_input.setPlaceholderText("Enter category")
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_search_term)
        train_button = QPushButton("Train Model")
        train_button.clicked.connect(self.train_model)
        train_layout.addWidget(self.term_input)
        train_layout.addWidget(self.category_input)
        train_layout.addWidget(add_button)
        train_layout.addWidget(train_button)
        layout.addLayout(train_layout)

        # Search section
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search query")
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.perform_search)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_button)
        layout.addLayout(search_layout)

        # Results section
        self.results_area = QTextEdit()
        self.results_area.setReadOnly(True)
        layout.addWidget(self.results_area)

        self.setLayout(layout)

    def add_search_term(self):
        term = self.term_input.text().strip()
        category = self.category_input.text().strip()
        if term and category:
            self.search_terms.append(term)
            self.categories.append(category)
            self.term_input.clear()
            self.category_input.clear()
            self.results_area.append(f"Added: {term} ({category})\n")
        else:
            QMessageBox.warning(self, "Input Error", "Please enter both a search term and a category.")

    def train_model(self):
        self.results_area.clear()
        self.results_area.append("Training model...\n")
        QApplication.processEvents()

        data = []
        for term, category in zip(self.search_terms, self.categories):
            results = self.search_google(term)
            if results and 'results' in results:
                for result in results['results']:
                    data.append((result.get('title', ''), result.get('description', ''), category))

        if not data:
            self.results_area.append("No data collected. Cannot train model.\n")
            return

        X = [f"{title} {description}" for title, description, _ in data]
        y = [category for _, _, category in data]

        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.clf = MultinomialNB()
        self.clf.fit(X_train, y_train)

        self.results_area.append("Model trained successfully!\n")

    def perform_search(self):
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Input Error", "Please enter a search query.")
            return
        if not self.clf or not self.vectorizer:
            QMessageBox.warning(self, "Model Error", "Please train the model first.")
            return

        results = self.search_google(query)
        if results and 'results' in results:
            self.results_area.clear()
            for result in results['results']:
                title = result.get('title', '')
                description = result.get('description', '')
                url = result.get('url', '')
                text = f"{title} {description}"
                vector = self.vectorizer.transform([text])
                prediction = self.clf.predict(vector)[0]
                
                self.results_area.append(f"Title: {title}")
                self.results_area.append(f"Category: {prediction}")
                self.results_area.append(f"URL: {url}")
                self.results_area.append("\n")
        else:
            self.results_area.append("No results found.\n")

    def search_google(self, query, limit=10):
        url = "https://google-search74.p.rapidapi.com/"
        headers = {
            "x-rapidapi-host": "google-search74.p.rapidapi.com",
            "x-rapidapi-key": "1447d2168dmsh60dde268103082ep155841jsncd7bb12af946"
        }
        querystring = {"query": query, "limit": str(limit), "related_keywords": "true"}
        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.results_area.append(f"API request failed: {str(e)}\n")
            return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SearchClassifierApp()
    ex.show()
    sys.exit(app.exec_())
