import polars as pl
from typing import List, Tuple
import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi
import re
from loguru import logger
from nltk.corpus import stopwords
import joblib
from pathlib import Path


class Preprocessor:
    def __init__(self, list_excel_file_path: List[str], list_NTP: List):
        self.list_excel_file_path = list_excel_file_path
        self.list_NTP = list_NTP
        self.df = None
        # save objects for encoding, scaling and bm25
        self.label_encoders = {}
        self.scalers = {}
        self.bm25_model = None
        self.vocabulary = None

    def _load_single_file(self, file_path: str, 
                         NTP: pl.datetime = pl.datetime(2016, 12, 27)) -> pl.DataFrame:
        # read excel file with file_path
        df = pl.read_excel(file_path, 
                           sheet_name="Sheet1",
                        #    read_options={"skip_rows": 4}
                           )
        # df = df[:, [2, 4, 6, 7]]    # Need to adjust based on actual file structure
        df.columns = ["Document No", "Title", "FA", "FC"]

        # datetime NTP = 27/12/2016
        # calculate days from NTP to FA and FC
        df = df.with_columns([
            (pl.col("FA") - NTP).dt.total_days().alias("NTP_to_FA"),
            (pl.col("FC") - NTP).dt.total_days().alias("NTP_to_FC")
        ])
        df = df.with_columns([
            (pl.col("NTP_to_FC") - pl.col("NTP_to_FA")).alias("FA_to_FC")
        ])
        # drop rows with FA_to_FC < 0
        df = df.filter(pl.col("FA_to_FC") >= 0)
        df = df.filter(pl.col("FA_to_FC") < 300)
        return df

    def load_data(self) -> pl.DataFrame:
        # load all excel files and concatenate them into a single dataframe
        df_list = []
        for file_path, NTP in zip(self.list_excel_file_path, self.list_NTP):
            df = self._load_single_file(file_path, NTP)
            df_list.append(df)
        df = pl.concat(df_list, how="vertical")
        self.df = df
        logger.info(f"Data loaded with shape: {df.shape}")
        return df

    def encode_document_no(self, df: pl.DataFrame) -> List[List[float]]:
        # Split Document No and extract parts 0, 1, 2
        doc_parts = df.select([
            pl.col("Document No").str.split("-").list.get(0).alias("part0"),
            pl.col("Document No").str.split("-").list.get(1).alias("part1"),
            pl.col("Document No").str.split("-").list.get(2).alias("part2")
        ])
        
        # Convert to numpy arrays for processing
        part0 = doc_parts["part0"].to_numpy()
        part1 = doc_parts["part1"].to_numpy()
        part2 = doc_parts["part2"].to_numpy()
        
        # Label encoding for each part
        le0, le1, le2 = LabelEncoder(), LabelEncoder(), LabelEncoder()
        encoded_part0 = le0.fit_transform(part0)
        encoded_part1 = le1.fit_transform(part1)
        encoded_part2 = le2.fit_transform(part2)
        # Save label encoders
        self.label_encoders = {
            'le0': le0,
            'le1': le1, 
            'le2': le2
        }
        
        # Get number of unique values for each part
        n_unique0 = len(le0.classes_)
        n_unique1 = len(le1.classes_)
        n_unique2 = len(le2.classes_)
        
        # Min-max scaling based on unique counts
        scaler0 = MinMaxScaler(feature_range=(0, 10))
        scaler1 = MinMaxScaler(feature_range=(0, 10))
        scaler2 = MinMaxScaler(feature_range=(0, 10))
        
        scaled_part0 = scaler0.fit_transform(encoded_part0.reshape(-1, 1) / (n_unique0 - 1)).flatten()
        scaled_part1 = scaler1.fit_transform(encoded_part1.reshape(-1, 1) / (n_unique1 - 1)).flatten()
        scaled_part2 = scaler2.fit_transform(encoded_part2.reshape(-1, 1) / (n_unique2 - 1)).flatten()
        # Save scalers
        self.scalers = {
            'scaler0': scaler0,
            'scaler1': scaler1,
            'scaler2': scaler2
        }

        # Create list of 3D vectors with pure Python floats
        vectors = []
        for i in range(len(df)):
            vector = [float(scaled_part0[i]), float(scaled_part1[i]), float(scaled_part2[i])]
            vectors.append(vector)
        
        return vectors

    def preprocess_titles(self) -> List[str]:
        """Preprocess titles by removing special characters and replacing with spaces"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get all titles and convert to list
        titles = self.df["Title"].to_list()
        
        # Clean each title by replacing special characters with spaces
        cleaned_titles = []
        for title in titles:
            if title is not None:
                # Replace special characters with spaces
                cleaned_title = re.sub(r'[\[\]()_+-]', ' ', str(title))
                # Replace multiple spaces with single space and strip
                cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
                cleaned_titles.append(cleaned_title)
            else:
                cleaned_titles.append("")
        
        return cleaned_titles

    def create_sparse_vectors(self) -> Tuple[List[List[float]], List[str]]:
        """Create BM25 sparse vectors using a more efficient approach"""
        # Get preprocessed titles
        titles = self.preprocess_titles()
        
        # Filter out empty titles and create corpus
        corpus = []
        valid_indices = []
        for i, title in enumerate(titles):
            if title.strip():
                corpus.append(title.split())
                valid_indices.append(i)
        
        if not corpus:
            logger.warning("No valid titles found for BM25 processing")
            return [], []
        
        # English stopwords from NLTK
        english_stopwords = set(stopwords.words('english'))
        
        # Count word occurrences across all documents
        word_counts = {}
        for doc in corpus:
            for word in doc:
                if word not in english_stopwords:  # Skip stopwords
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Build vocabulary from corpus - only words appearing at least twice and not stopwords
        vocabulary = [word for word, count in word_counts.items() if count >= 2]
        vocabulary.sort()  # Sort for consistent ordering
        print(f"Vocabulary: {vocabulary[:100]}")
        
        # Create BM25 model with corpus
        bm25 = BM25Okapi(corpus)
        self.bm25_model = bm25
        self.vocabulary = vocabulary

        # Generate sparse vectors (using actual sparse representation)
        sparse_vectors = [[] for _ in titles]
        
        # For each document, score it against each term in vocabulary
        for term_idx, term in enumerate(vocabulary):
            # Create a query with just this term
            query = [term]
            # Get scores for all documents for this term
            scores = bm25.get_scores(query)
            
            # Add non-zero scores to sparse vectors
            for doc_idx, score in enumerate(scores):
                if score > 0:
                    original_idx = valid_indices[doc_idx]
                    # Store as (term_idx, score) for sparse representation
                    if not sparse_vectors[original_idx]:
                        sparse_vectors[original_idx] = []
                    sparse_vectors[original_idx].append((term_idx, float(score)))
        
        # Convert sparse representation to dense if needed
        dense_vectors = []
        for vec in sparse_vectors:
            dense_vec = [0.0] * len(vocabulary)
            for term_idx, score in vec:
                dense_vec[term_idx] = score
            dense_vectors.append(dense_vec)
        
        return dense_vectors
    
    def generate_embedding_vectors(self):
        """Generate embedding vectors by combining Document No and Title embeddings"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Encode Document No
        doc_no_vectors = self.encode_document_no(self.df)
        
        # Create sparse vectors from Titles
        title_vectors = self.create_sparse_vectors()
        
        if len(doc_no_vectors) != len(title_vectors):
            raise ValueError("Mismatch in number of Document No vectors and Title vectors")
        
        # Combine both vectors
        combined_vectors = []
        for doc_vec, title_vec in zip(doc_no_vectors, title_vectors):
            combined_vec = doc_vec + title_vec
            combined_vectors.append(combined_vec)
        
        # update 3 columns no_embed, title_embed, concat_embed to self.df
        self.df = self.df.with_columns([
            pl.Series("no_embed", doc_no_vectors),
            pl.Series("title_embed", title_vectors),
            pl.Series("concat_embed", combined_vectors)
        ])

        return self.df
    
    def save_fitted_models(self, save_dir: str = "../models"):
        """Save models to disk"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save label encoders
        if self.label_encoders:
            joblib.dump(self.label_encoders, save_path / "label_encoders.pkl")
            logger.info(f"Label encoders saved to {save_path / 'label_encoders.pkl'}")
        
        # Save scalers  
        if self.scalers:
            joblib.dump(self.scalers, save_path / "scalers.pkl")
            logger.info(f"Scalers saved to {save_path / 'scalers.pkl'}")
        
        # Save BM25 model and vocabulary
        if self.bm25_model and self.vocabulary:
            bm25_data = {
                'model': self.bm25_model,
                'vocabulary': self.vocabulary
            }
            joblib.dump(bm25_data, save_path / "bm25_model.pkl")
            logger.info(f"BM25 model saved to {save_path / 'bm25_model.pkl'}")

    def load_fitted_models(self, save_dir: str = "../models"):
        """Load models from disk"""
        save_path = Path(save_dir)
        
        # Load label encoders
        if (save_path / "label_encoders.pkl").exists():
            self.label_encoders = joblib.load(save_path / "label_encoders.pkl")
            logger.info("Label encoders loaded successfully")
        
        # Load scalers
        if (save_path / "scalers.pkl").exists():
            self.scalers = joblib.load(save_path / "scalers.pkl")
            logger.info("Scalers loaded successfully")
        
        # Load BM25 model
        if (save_path / "bm25_model.pkl").exists():
            bm25_data = joblib.load(save_path / "bm25_model.pkl")
            self.bm25_model = bm25_data['model']
            self.vocabulary = bm25_data['vocabulary']
            logger.info("BM25 model loaded successfully")

    def preprocess(self, save_dir: str = "../models"):
        """Preprocess data using fitted models"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Load fitted models
        self.load_fitted_models(save_dir)
        
        # Encode document numbers using fitted encoders and scalers
        doc_parts = self.df.select([
            pl.col("Document No").str.split("-").list.get(0).alias("part0"),
            pl.col("Document No").str.split("-").list.get(1).alias("part1"),
            pl.col("Document No").str.split("-").list.get(2).alias("part2")
        ])
        
        part0 = doc_parts["part0"].to_numpy()
        part1 = doc_parts["part1"].to_numpy()
        part2 = doc_parts["part2"].to_numpy()
        
        # Transform using fitted encoders
        encoded_part0 = self.label_encoders['le0'].transform(part0)
        encoded_part1 = self.label_encoders['le1'].transform(part1)
        encoded_part2 = self.label_encoders['le2'].transform(part2)
        
        # Get unique counts from fitted encoders
        n_unique0 = len(self.label_encoders['le0'].classes_)
        n_unique1 = len(self.label_encoders['le1'].classes_)
        n_unique2 = len(self.label_encoders['le2'].classes_)
        
        # Transform using fitted scalers
        scaled_part0 = self.scalers['scaler0'].transform((encoded_part0.reshape(-1, 1) / (n_unique0 - 1))).flatten()
        scaled_part1 = self.scalers['scaler1'].transform((encoded_part1.reshape(-1, 1) / (n_unique1 - 1))).flatten()
        scaled_part2 = self.scalers['scaler2'].transform((encoded_part2.reshape(-1, 1) / (n_unique2 - 1))).flatten()
        
        # Create document vectors
        doc_no_vectors = []
        for i in range(len(self.df)):
            vector = [float(scaled_part0[i]), float(scaled_part1[i]), float(scaled_part2[i])]
            doc_no_vectors.append(vector)
        
        # Create title vectors using fitted BM25 model
        titles = self.preprocess_titles()
        title_vectors = []
        
        for title in titles:
            dense_vec = [0.0] * len(self.vocabulary)
            if title.strip():
                words = title.split()
                for term_idx, term in enumerate(self.vocabulary):
                    if term in words:
                        score = self.bm25_model.get_scores([term])
                        if len(score) > 0:
                            dense_vec[term_idx] = float(score[0])
            title_vectors.append(dense_vec)
        
        # Combine vectors
        combined_vectors = []
        for doc_vec, title_vec in zip(doc_no_vectors, title_vectors):
            combined_vec = doc_vec + title_vec
            combined_vectors.append(combined_vec)
        
        # Update dataframe with embedding columns
        self.df = self.df.with_columns([
            pl.Series("no_embed", doc_no_vectors),
            pl.Series("title_embed", title_vectors),
            pl.Series("concat_embed", combined_vectors)
        ])
        
        return self.df

def main(excel_file_path: str):
    """Test function for encode_document_no and create_sparse_vectors methods"""
    try:
        # Create preprocessor instance with actual file path
        preprocessor = Preprocessor([excel_file_path], [pl.datetime(2016, 12, 27)])
        
        # Load data from Excel file
        logger.info(f"Loading data from: {excel_file_path}")
        df = preprocessor.load_data()
        logger.info(f"Loaded {len(df)} rows of data")
        print("-" * 40)

        logger.info("Testing encode_document_no method:")
        print("-" * 40)
        encoded_vectors = preprocessor.encode_document_no(df)
        for i, (doc_no, vector) in enumerate(zip(df["Document No"].to_list()[:5], encoded_vectors[:5])):
            print(f"{doc_no}: {vector}")
        print(f"... and {len(encoded_vectors) - 5} more vectors")

        logger.info("Testing create_sparse_vectors method:")
        sparse_vectors = preprocessor.create_sparse_vectors()
        for i, (title, vector) in enumerate(zip(df["Title"].to_list()[:3], sparse_vectors[:3])):
            print(f"Title: {title}")
            print(f"Sparse vector length: {len(vector)}")
            print(f"Non-zero values: {sum(1 for x in vector if x > 0)}")
            print(f"Sample non-zero values: {[x for x in vector if x > 0][:5]}")
            print()

        logger.info("Testing generate_embedding_vectors method:")
        data = preprocessor.generate_embedding_vectors()
        print(data.head(3))

        logger.info("Saving fitted models...")
        preprocessor.save_fitted_models()
        logger.info("All fitted models saved successfully.")

    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    excel_file_path = "../data/GratiMDL.xlsx"
    main(excel_file_path)

