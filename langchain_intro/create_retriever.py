import dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEW_CHROMA_PATH = "chroma-data"

dotenv.load_dotenv()

loader = CSVLoader(file_path = REVIEWS_CSV_PATH, source_column = "review")
reviews = loader.load()

reviews_vector_db = Chroma.from_documents(
    reviews, OpenAIEmbeddings(), persist_directory=REVIEW_CHROMA_PATH
)