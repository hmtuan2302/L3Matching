from __future__ import annotations

import numpy as np
import polars as pl
from typing import Tuple, Optional, List, Dict, Any
from datetime import datetime, timedelta
from shared.utils import get_logger, profile
from shared.base import BaseModel
from domain.preprocess.embedding import EmbeddingProcessor
from domain.date_generation.match_title import fuzzy_search, exact_match, find_similar_embeddings, build_hnsw_index
from domain.date_generation.metrics import iou_mean, mae

logger = get_logger(__name__)


class DateGenerationInput(BaseModel):
    """Input for date generation processing."""
    model_config = {"arbitrary_types_allowed": True}
    document_title: str
    threshold_number: int = 10
    input_data: pl.DataFrame
    hist_data: pl.DataFrame
    fuzzy_threshold: float = 0.8
    embedding_threshold: float = 0.7
    plan_date: Optional[datetime] = None
    plan_final_date: Optional[datetime] = None
    actual_first_date: Optional[datetime] = None
    actual_final_date: Optional[datetime] = None

class DateGenerationOutput(BaseModel):
    """Output from date generation processing."""
    model_config = {"arbitrary_types_allowed": True}
    output_df: pl.DataFrame
    iou: Optional[float] = None
    mae_first_date: Optional[float] = None
    mae_final_date: Optional[float] = None

class DateGenerationProcessor:
    """Processor for generating date predictions based on historical data."""

    @profile
    def process(
        self,
        input_df: pl.DataFrame,
        hist_data: pl.DataFrame,
        threshold_number: int = 10,
        fuzzy_threshold: float = 0.8,
        embedding_threshold: float = 0.7
    ) -> DateGenerationOutput:
        """Batch date generation pipeline for all rows in input_df."""
        results = []

        for row in input_df.iter_rows(named=True):
            logger.info(f"Processing document: {row.get('Title', 'Unknown Title')}")
            document_title = row.get('Title')
            plan_date = row.get('Plan - First Date')
            plan_final_date = row.get('Plan - Final Date')
            actual_first_date = row.get('Actual - First Date')
            actual_final_date = row.get('Actual - Final Date')

            logger.info(f"Parsed dates - Plan: {plan_date}, Plan Final: {plan_final_date}, Actual First: {actual_first_date}, Actual Final: {actual_final_date}")
            if not document_title or not plan_date:
                continue

            predicted_first_date_days, predicted_final_date_days, calculation_method, explanations = self.date_generation(
                document_title=document_title,
                thres_no=threshold_number,
                hist_data=hist_data,
                fuzzy_threshold=fuzzy_threshold,
                embedding_threshold=embedding_threshold,
                input_data=input_df
            )

            predicted_first_date_actual = (
                plan_date + timedelta(days=predicted_first_date_days)
                if plan_date and predicted_first_date_days is not None else None
            )
            predicted_final_date_actual = (
                plan_final_date + timedelta(days=predicted_final_date_days)
                if plan_final_date and predicted_final_date_days is not None else None
            )

            output_data = {
                "Document Title": document_title,
                "Plan - First Date": plan_date,
                "Plan - Final Date": plan_final_date,
                "Actual - First Date": actual_first_date,
                "Actual - Final Date": actual_final_date,
                "Predicted - First Date (Days)": predicted_first_date_days,
                "Predicted - Final Date (Days)": predicted_final_date_days,
                "Predict - First Date": predicted_first_date_actual,
                "Predict - Final Date": predicted_final_date_actual,
                "calculation_method": calculation_method,
                "explanations": explanations
            }
            results.append(output_data)

        output_df = pl.DataFrame(results)
        # --- Metrics calculation ---
        # Only calculate metrics if actual dates are available for all rows
        pred_ranges = []
        actual_ranges = []
        pred_first_dates = []
        actual_first_dates = []
        pred_final_dates = []
        actual_final_dates = []

        for row in output_df.iter_rows(named=True):
            pred_first = row.get("Predict - First Date")
            pred_final = row.get("Predict - Final Date")
            act_first = row.get("Actual - First Date")
            act_final = row.get("Actual - Final Date")
            # Only include if all dates are present
            if pred_first and pred_final and act_first and act_final:
                pred_ranges.append((pred_first, pred_final))
                actual_ranges.append((act_first, act_final))
                pred_first_dates.append(pred_first)
                actual_first_dates.append(act_first)
                pred_final_dates.append(pred_final)
                actual_final_dates.append(act_final)

        iou = iou_mean(pred_ranges, actual_ranges) if pred_ranges else None
        mae_first_date = mae(pred_first_dates, actual_first_dates) if pred_first_dates else None
        mae_final_date = mae(pred_final_dates, actual_final_dates) if pred_final_dates else None

        return DateGenerationOutput(
            output_df=output_df,
            iou=iou,
            mae_first_date=mae_first_date,
            mae_final_date=mae_final_date
        )
    
    def date_generation(
        self, 
        document_title: str, 
        thres_no: int, 
        hist_data: pl.DataFrame,
        fuzzy_threshold: float = 0.8,
        embedding_threshold: float = 0.7,
        input_data: Optional[pl.DataFrame] = None
    ) -> Tuple[Optional[float], Optional[float], Optional[str], List[Dict[str, Any]]]:
        """
        Generate date predictions based on historical data.
        
        Logic:
        1. Try exact match first
        2. If exact match has n >= 3, use it directly
        3. If exact match has n < 3, use all 3 methods
        
        Args:
            document_title: Title of the document to predict dates for
            thres_no: Threshold number of documents required for average calculation (default 5)
            hist_data: Historical data DataFrame with Range columns
            fuzzy_threshold: Threshold for fuzzy matching (0-1)
            embedding_threshold: Threshold for embedding similarity (0-1)
            input_data: Input data DataFrame (for precomputed embeddings optimization)
            
        Returns:
            Tuple: (predicted_first_date_in_days, predicted_final_date_in_days, calculation_method, explanations)
        """
        try:
            # Stage 1: Try exact match first
            exact_matches = self.filter_hist_document(document_title, hist_data)
            exact_count = exact_matches.height
            
            logger.info(f"Stage 1 - Exact matches found: {exact_count} for title: {document_title}")
            
            if exact_count >= 3:
                # Use only exact matches
                final_data = exact_matches
                explanations = []
                
                # Create explanations for exact matches
                for row in exact_matches.iter_rows(named=True):
                    explanation = {
                        "document_title": row.get('Title', ''),
                        "Range - First Date": row.get('Range - First Date', 0.0),
                        "Range - Final Date": row.get('Range - Final Date', 0.0),
                        "matching_method": "exact_match",
                        "confidence_score": "100%"
                    }
                    explanations.append(explanation)
                    
                no_doc = exact_count
                logger.info(f"Using exact matches only: {no_doc} documents")
                
            else:
                # Stage 2: Use all 3 methods (exact + fuzzy + semantic)
                logger.info(f"Stage 2 - Exact matches insufficient ({exact_count} < 3), using all methods")
                
                matched_data, match_explanations = self.find_all_matches_with_explanations(
                    document_title, hist_data, fuzzy_threshold, embedding_threshold, input_data
                )
                
                # Remove duplicates while preserving match type information
                final_data, explanations = self.remove_duplicates_preserve_matches(
                    matched_data, match_explanations
                )
                
                no_doc = final_data.height
                logger.info(f"After all methods and deduplication: {no_doc} documents")
            
            # Determine calculation method based on final number of documents
            if no_doc >= thres_no:
                # Sufficient data for average calculation
                predicted_first, predicted_final = self.calculate_average_hist(final_data)
                calculation_method = "average_calculation"
                
            elif 3 <= no_doc < thres_no:
                # Use Monte Carlo simulation
                predicted_first, predicted_final = self.simulate_monte_carlo(final_data)
                calculation_method = "monte_carlo_simulation"
                
            elif 1 <= no_doc < 3:
                # Limited data average
                predicted_first, predicted_final = self.calculate_average_hist(final_data)
                calculation_method = "limited_average"
                
            else:
                # No documents found - return None for dates
                logger.warning(f"No relevant documents found for title: {document_title}")
                return None, None, None, [{
                    "document_title": "No matches found",
                    "Range - First Date": None,
                    "Range - Final Date": None,
                    "matching_method": "none",
                    "confidence_score": "0%"
                }]
            
            return predicted_first, predicted_final, calculation_method, explanations
            
        except Exception as e:
            logger.error(f"Error in date generation: {str(e)}")
            return None, None, "error_fallback", [{
                "document_title": f"Error: {str(e)}",
                "Range - First Date": None,
                "Range - Final Date": None,
                "matching_method": "error",
                "confidence_score": "0%"
            }]

    def find_all_matches_with_explanations(
        self, 
        document_title: str, 
        hist_data: pl.DataFrame,
        fuzzy_threshold: float,
        embedding_threshold: float,
        input_data: Optional[pl.DataFrame] = None
    ) -> Tuple[pl.DataFrame, List[Dict[str, Any]]]:
        """Find all matches using all 3 methods and create explanations for each."""
        all_matches = []
        all_explanations = []
        
        # 1. Exact matches (highest priority) - using exact_match function
        exact_matches = self.find_exact_matches(document_title, hist_data)
        if exact_matches.height > 0:
            all_matches.append(exact_matches)
            
            # Create explanations for exact matches
            for row in exact_matches.iter_rows(named=True):
                explanation = {
                    "document_title": row.get('Title', ''),
                    "Range - First Date": row.get('Range - First Date', 0.0),
                    "Range - Final Date": row.get('Range - Final Date', 0.0),
                    "matching_method": "exact_match",
                    "confidence_score": "100%"
                }
                all_explanations.append(explanation)
        
        # 2. Fuzzy matches (medium priority) - using fuzzy_search function
        fuzzy_matches, fuzzy_scores = self.find_fuzzy_similar_titles_with_scores(
            document_title, hist_data, fuzzy_threshold
        )
        if fuzzy_matches.height > 0:
            all_matches.append(fuzzy_matches)
            
            # Create explanations for fuzzy matches
            for i, row in enumerate(fuzzy_matches.iter_rows(named=True)):
                score = fuzzy_scores[i] if i < len(fuzzy_scores) else fuzzy_threshold * 100
                explanation = {
                    "document_title": row.get('Title', ''),
                    "Range - First Date": row.get('Range - First Date', 0.0),
                    "Range - Final Date": row.get('Range - Final Date', 0.0),
                    "matching_method": "fuzzy_match",
                    "confidence_score": f"{score:.1f}%"  # fuzzy_search returns percentage
                }
                all_explanations.append(explanation)
        
        # 3. Embedding similarity matches (lowest priority) - OPTIMIZED VERSION
        if 'Title_Embedding' in hist_data.columns:
            embedding_matches, embedding_scores = self.find_embedding_similar_titles_with_precomputed(
                document_title, hist_data, embedding_threshold, input_data
            )
            if embedding_matches.height > 0:
                all_matches.append(embedding_matches)
                
                # Create explanations for embedding matches
                for i, row in enumerate(embedding_matches.iter_rows(named=True)):
                    score = embedding_scores[i] if i < len(embedding_scores) else embedding_threshold
                    explanation = {
                        "document_title": row.get('Title', ''),
                        "Range - First Date": row.get('Range - First Date', 0.0),
                        "Range - Final Date": row.get('Range - Final Date', 0.0),
                        "matching_method": "semantic_match",
                        "confidence_score": f"{score*100:.1f}%"
                    }
                    all_explanations.append(explanation)
        
        # Combine all matches
        if all_matches:
            combined_data = pl.concat(all_matches)
            return combined_data, all_explanations
        else:
            return pl.DataFrame(), []

    def find_exact_matches(self, document_title: str, hist_data: pl.DataFrame) -> pl.DataFrame:
        """Find exact matches using the exact_match function from match_title."""
        try:
            titles = hist_data.select(pl.col('Title')).to_series().to_list()
            
            # Use exact_match function from match_title
            exact_titles = exact_match(titles, document_title)
            
            if exact_titles:
                # Filter historical data to get matching rows
                filtered = hist_data.filter(
                    pl.col('Title').is_in(exact_titles)
                )
                logger.info(f"Found {filtered.height} exact matches for: {document_title}")
                return filtered
            else:
                return pl.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in exact matching: {str(e)}")
            return pl.DataFrame()

    def find_fuzzy_similar_titles_with_scores(
        self, 
        document_title: str, 
        hist_data: pl.DataFrame, 
        threshold: float = 0.8
    ) -> Tuple[pl.DataFrame, List[float]]:
        """Find titles using fuzzy_search function from match_title."""
        try:
            titles = hist_data.select(pl.col('Title')).to_series().to_list()
            
            # Convert threshold from 0-1 to 0-100 for fuzzy_search
            threshold_percentage = int(threshold * 100)
            
            # Use fuzzy_search function from match_title
            fuzzy_matches = fuzzy_search(
                input_title=document_title,
                title_list=titles,
                limit=50,  # Get more results to filter later
                threshold=threshold_percentage
            )
            
            matches = []
            scores = []
            
            for matched_title, score in fuzzy_matches:
                # Find rows in dataframe that match this title
                matching_rows = hist_data.filter(pl.col('Title') == matched_title)
                if matching_rows.height > 0:
                    matches.append(matching_rows)
                    scores.append(float(score))  # fuzzy_search returns score as percentage
            
            if matches:
                combined_matches = pl.concat(matches)
                logger.info(f"Found {combined_matches.height} fuzzy matches for: {document_title}")
                return combined_matches, scores
            else:
                return pl.DataFrame(), []
                
        except Exception as e:
            logger.error(f"Error in fuzzy matching: {str(e)}")
            return pl.DataFrame(), []

    def find_embedding_similar_titles_with_precomputed(
        self, 
        document_title: str, 
        hist_data: pl.DataFrame, 
        threshold: float = 0.7,
        input_data: Optional[pl.DataFrame] = None
    ) -> Tuple[pl.DataFrame, List[float]]:
        """OPTIMIZED: Find titles using precomputed embeddings when available."""
        try:
            if 'Title_Embedding' not in hist_data.columns:
                logger.warning("No Title_Embedding column found in historical data")
                return pl.DataFrame(), []
            
            embedding_processor = EmbeddingProcessor()
            input_embedding = None
            
            # OPTIMIZATION 1: Try to get precomputed embedding from input_data first
            if input_data is not None and 'Title_Embedding' in input_data.columns:
                try:
                    # Look for exact title match in input_data
                    matching_input = input_data.filter(pl.col('Title') == document_title)
                    
                    if matching_input.height > 0:
                        embedding_str = matching_input.select('Title_Embedding').item()
                        if embedding_str and str(embedding_str).strip() != '' and str(embedding_str) != 'None':
                            try:
                                input_embedding = np.array(
                                    embedding_processor.string_to_embedding(embedding_str),
                                    dtype='float32'
                                )
                                logger.info(f"✅ Using precomputed embedding for: {document_title}")
                            except Exception as e:
                                logger.warning(f"Failed to parse precomputed embedding: {e}")
                                input_embedding = None
                    
                    # OPTIMIZATION 2: If exact match not found, try partial matching in input_data
                    if input_embedding is None:
                        partial_matches = input_data.filter(
                            pl.col('Title').str.to_lowercase().str.contains(document_title.lower(), literal=False)
                        )
                        
                        if partial_matches.height > 0:
                            # Use the first partial match
                            embedding_str = partial_matches.select('Title_Embedding').item()
                            if embedding_str and str(embedding_str).strip() != '' and str(embedding_str) != 'None':
                                try:
                                    input_embedding = np.array(
                                        embedding_processor.string_to_embedding(embedding_str),
                                        dtype='float32'
                                    )
                                    matched_title = partial_matches.select('Title').item()
                                    logger.info(f"✅ Using precomputed embedding from similar title: {matched_title}")
                                except Exception as e:
                                    logger.warning(f"Failed to parse partial match embedding: {e}")
                                    input_embedding = None
                
                except Exception as e:
                    logger.warning(f"Error looking up precomputed embedding: {e}")
                    input_embedding = None
            
            # FALLBACK: Generate embedding if not found in precomputed data
            if input_embedding is None:
                logger.info(f"🔄 Generating new embedding for: {document_title} (no precomputed found)")
                try:
                    input_embedding = embedding_processor.generate_embeddings([document_title])[0]
                    input_embedding = np.array(input_embedding, dtype='float32')
                except Exception as e:
                    logger.error(f"Failed to generate embedding: {e}")
                    return pl.DataFrame(), []
            
            # Process historical embeddings (same as before)
            embedding_strings = hist_data.select(pl.col('Title_Embedding')).to_series().to_list()
            historical_embeddings = []
            valid_indices = []
            
            for i, emb_str in enumerate(embedding_strings):
                if emb_str and str(emb_str).strip() != '' and str(emb_str) != 'None':
                    try:
                        hist_embedding = np.array(
                            embedding_processor.string_to_embedding(emb_str),
                            dtype='float32'
                        )
                        historical_embeddings.append(hist_embedding)
                        valid_indices.append(i)
                    except Exception:
                        continue
            
            if not historical_embeddings:
                logger.warning("No valid historical embeddings found")
                return pl.DataFrame(), []
            
            historical_embeddings = np.array(historical_embeddings)
            
            # Use find_similar_embeddings function from match_title
            similar_results = find_similar_embeddings(
                input_embedding=input_embedding,
                historical_embeddings=historical_embeddings,
                limit=50,
                threshold=threshold,
                include_self=False
            )
            
            matches = []
            scores = []
            
            for result in similar_results:
                embedding_idx = result['index']
                similarity = result['similarity']
                
                # Map back to original dataframe index
                original_idx = valid_indices[embedding_idx]
                matching_row = hist_data.slice(original_idx, 1)
                
                matches.append(matching_row)
                scores.append(similarity)
            
            if matches:
                combined_matches = pl.concat(matches)
                logger.info(f"Found {combined_matches.height} embedding matches for: {document_title}")
                return combined_matches, scores
            else:
                return pl.DataFrame(), []
                
        except Exception as e:
            logger.error(f"Error in embedding similarity: {str(e)}")
            return pl.DataFrame(), []

    def find_embedding_similar_titles_with_scores(
        self, 
        document_title: str, 
        hist_data: pl.DataFrame, 
        threshold: float = 0.7
    ) -> Tuple[pl.DataFrame, List[float]]:
        """DEPRECATED: Use find_embedding_similar_titles_with_precomputed instead for optimization."""
        logger.warning("Using deprecated method. Consider using find_embedding_similar_titles_with_precomputed for better performance.")
        return self.find_embedding_similar_titles_with_precomputed(document_title, hist_data, threshold, None)

    def remove_duplicates_preserve_matches(
        self, 
        data: pl.DataFrame, 
        explanations: List[Dict[str, Any]]
    ) -> Tuple[pl.DataFrame, List[Dict[str, Any]]]:
        """
        Remove duplicate rows while preserving the best match type based on priority.
        Priority: exact_match > fuzzy_match > semantic_match
        """
        if data.height == 0:
            return data, explanations
        
        # Create a mapping of explanations to data rows
        explanation_map = {i: explanations[i] for i in range(len(explanations))}
        
        # Priority order: exact_match > fuzzy_match > semantic_match
        method_priority = {
            "exact_match": 3, 
            "fuzzy_match": 2, 
            "semantic_match": 1,
            "none": 0,
            "error": 0
        }
        
        if 'Document No' in data.columns:
            # Group by Document No and keep the best match type
            unique_docs = {}
            unique_explanations_dict = {}
            
            for i, row in enumerate(data.iter_rows(named=True)):
                doc_no = row.get('Document No', f'row_{i}')
                current_method = explanation_map[i]['matching_method']
                current_priority = method_priority.get(current_method, 0)
                
                if doc_no not in unique_docs:
                    unique_docs[doc_no] = (i, row)
                    unique_explanations_dict[doc_no] = explanation_map[i]
                else:
                    existing_priority = method_priority.get(
                        unique_explanations_dict[doc_no]['matching_method'], 0
                    )
                    if current_priority > existing_priority:
                        # Replace with higher priority match
                        unique_docs[doc_no] = (i, row)
                        unique_explanations_dict[doc_no] = explanation_map[i]
                        logger.debug(f"Replaced {doc_no}: {unique_explanations_dict[doc_no]['matching_method']} -> {current_method}")
                    elif current_priority == existing_priority:
                        # If same priority, keep the one with higher confidence
                        current_conf = self._extract_confidence_value(explanation_map[i]['confidence_score'])
                        existing_conf = self._extract_confidence_value(unique_explanations_dict[doc_no]['confidence_score'])
                        if current_conf > existing_conf:
                            unique_docs[doc_no] = (i, row)
                            unique_explanations_dict[doc_no] = explanation_map[i]
            
            # Reconstruct DataFrame and explanations
            unique_rows = [row_data for _, row_data in unique_docs.values()]
            unique_explanations = list(unique_explanations_dict.values())
            
            if unique_rows:
                unique_data = pl.DataFrame(unique_rows)
                return unique_data, unique_explanations
        
        # Fallback: use original data if no Document No column
        return data.unique(), explanations

    def _extract_confidence_value(self, confidence_str: str) -> float:
        """Extract numerical confidence value from string."""
        try:
            if confidence_str == "N/A" or confidence_str is None:
                return 0.0
            return float(str(confidence_str).replace('%', ''))
        except:
            return 0.0

    def calculate_average_hist(self, hist_document: pl.DataFrame) -> Tuple[float, float]:
        """
        Calculate average lead times using Range columns from preprocessing.

        Returns:
            predicted_first: Average of Range - First Date in days
            predicted_final: Average of Range - Final Date in days
        """
        try:
            # Check if Range columns exist (from preprocessing pipeline)
            if 'Range - First Date' not in hist_document.columns or 'Range - Final Date' not in hist_document.columns:
                logger.warning("Range columns not found. Data may not be preprocessed.")
                return 0.0, 0.0
            
            # Calculate averages, handling null values
            predicted_first = hist_document.select(
                pl.col("Range - First Date").filter(pl.col("Range - First Date").is_not_null()).mean()
            ).item() or 0.0
            
            predicted_final = hist_document.select(
                pl.col("Range - Final Date").filter(pl.col("Range - Final Date").is_not_null()).mean()
            ).item() or 0.0

            logger.info(f"Calculated averages: First={predicted_first:.2f}, Final={predicted_final:.2f}")
            return predicted_first, predicted_final
            
        except Exception as e:
            logger.error(f"Error calculating averages: {str(e)}")
            return 0.0, 0.0

    def simulate_monte_carlo(
        self, 
        hist_document: pl.DataFrame, 
        n_samples: int = 20000, 
        percentile_policy: int = 80
    ) -> Tuple[float, float]:
        """
        Simulate date differences using Monte Carlo sampling with Range columns.

        Args:
            hist_document: DataFrame with Range - First Date and Range - Final Date columns
            n_samples: Number of Monte Carlo samples
            percentile_policy: Discipline policy (e.g., 50, 80, 90)

        Returns:
            predicted_first, predicted_final based on sampled percentile
        """
        try:
            results = {}
            
            range_columns = {
                "Range - First Date": "first", 
                "Range - Final Date": "final"
            }
            
            for col, key in range_columns.items():
                if col not in hist_document.columns:
                    logger.warning(f"Column {col} not found")
                    results[key] = 0.0
                    continue
                
                # Extract non-null values
                data = hist_document.select(
                    pl.col(col).filter(pl.col(col).is_not_null())
                ).to_numpy().flatten()
                
                if len(data) == 0:
                    results[key] = 0.0
                    continue

                # Handle negative values by shifting
                shift = 0
                if np.any(data <= 0):
                    shift = abs(np.min(data)) + 1
                    data_shifted = data + shift
                else:
                    data_shifted = data

                # Fit log-normal distribution (estimate parameters)
                mu = np.mean(np.log(data_shifted))
                sigma = np.std(np.log(data_shifted))

                # Monte Carlo sampling
                samples = np.random.lognormal(mean=mu, sigma=sigma, size=n_samples)

                # Shift back if shifted
                if shift != 0:
                    samples = samples - shift

                # Apply percentile policy
                results[key] = np.percentile(samples, percentile_policy)

            return results.get("first", 0.0), results.get("final", 0.0)
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return 0.0, 0.0

    def filter_hist_document(self, document_title: str, hist_data: pl.DataFrame) -> pl.DataFrame:
        """Apply partial match (case insensitive) to find historical data of a document."""
        try:
            # Filter using case-insensitive partial matching
            filtered = hist_data.filter(
                pl.col('Title').str.to_lowercase().str.contains(
                    document_title.lower(), literal=False
                )
            )
            logger.info(f"Filtered {filtered.height} documents for exact match: {document_title}")
            return filtered
            
        except Exception as e:
            logger.error(f"Error filtering historical documents: {str(e)}")
            return pl.DataFrame()
