#!/usr/bin/env python3
"""
Automatic Problem Type Predictor and Error Checker
Uses the trained model to predict problem types on unseen datasets and identifies potential human labeling errors.
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

class ProblemTypePredictor:
    """
    Automatic problem type predictor that can validate human classifications.
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the predictor."""
        self.config_path = config_path
        self.load_config()
        self.load_trained_components()
        
        # Thresholds for error detection
        # Model must be confident to flag errors
        self.confidence_threshold = 0.75  
        # Minimum confidence difference to flag
        self.disagreement_threshold = 0.40  
        
    def load_config(self):
        """Load configuration."""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            print("Configuration loaded successfully")
        except Exception as e:
            print(f"Error loading config: {e}")
            raise
    
    def load_trained_components(self):
        """Load all trained model components."""
        print("Loading trained model components...")
        
        try:
            # Load model
            from tensorflow.keras.models import load_model
            model_path = os.path.join(self.config['paths']['models'], 'best_model.h5')
            if not os.path.exists(model_path):
                model_path = os.path.join(self.config['paths']['models'], 'text_classification_model.h5')
            
            self.model = load_model(model_path)
            print(f"Model loaded from: {model_path}")
            
            # Load tokenizer
            tokenizer_path = os.path.join(self.config['paths']['processed_data'], 'tokenizer.pkl')
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print("Tokenizer loaded successfully")
            
            # Load label encoder
            encoder_path = os.path.join(self.config['paths']['processed_data'], 'label_encoder.pkl')
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("Label encoder loaded successfully")
            
            # Load data preprocessor
            from src.data_preprocessing import DataPreprocessor
            self.preprocessor = DataPreprocessor(self.config_path)
            print("Data preprocessor initialized")
            
            # Get model parameters
            self.max_length = self.config['model'].get('max_sequence_length', 100)
            self.vocab_size = len(self.tokenizer.word_index) + 1
            
            print(f"Model parameters:")
            print(f"  - Max sequence length: {self.max_length}")
            print(f"  - Vocabulary size: {self.vocab_size}")
            print(f"  - Number of classes: {len(self.label_encoder.classes_)}")
            print(f"  - Classes: {', '.join(self.label_encoder.classes_[:10])}")
            if len(self.label_encoder.classes_) > 10:
                print(f"    ... and {len(self.label_encoder.classes_) - 10} more")
                
        except Exception as e:
            print(f"Error loading trained components: {e}")
            print("Make sure the model has been trained")
            raise
    
    def preprocess_descriptions(self, descriptions):
        """Preprocess descriptions for model input."""

        # Clean text using the same preprocessing as training
        cleaned_text = []
        for desc in descriptions:
            cleaned = self.preprocessor.clean_text(str(desc))
            cleaned_text.append(cleaned)
        
        # Tokenize and pad sequence it
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except ImportError:
            from keras.preprocessing.sequence import pad_sequences
        
        sequences = self.tokenizer.texts_to_sequences(cleaned_text)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        print(f"Text preprocessed successfully")
        return padded_sequences, cleaned_text
    
    def predict_problem_types(self, X_processed):
        """Make predictions using the trained model."""
        print("Predicting using the trained model")
        
        # Get predictions
        predictions = self.model.predict(X_processed, verbose=0)
        
        # Get predicted classes and confidence scores
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        # Convert to labels
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        
        print(f"Predictions completed for {len(predicted_labels)} samples")
        
        return predicted_labels, confidence_scores, predictions
    
    def analyze_predictions(self, df, predicted_labels, confidence_scores, all_predictions, 
                          description_col='description', true_label_col='problem_type'):
        """Analyze predictions to identify potential human labeling errors."""
        print("Analyzing predictions")
        
        # Create results dataframe
        results_df = df.copy()
        results_df['model_prediction'] = predicted_labels
        results_df['model_confidence'] = confidence_scores
        results_df['prediction_matches'] = results_df[true_label_col] == results_df['model_prediction']
        
        # Calculate confidence
        manual_label_confidences = []
        confidence_differences = []
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            manual_label = row[true_label_col]
            
            # Find confidence for manual-provided label
            try:
                manual_label_idx = list(self.label_encoder.classes_).index(manual_label)
                manual_confidence = all_predictions[i][manual_label_idx]
            except ValueError:
                # Manual label not in training data
                manual_confidence = 0.0

            manual_label_confidences.append(manual_confidence)

            # Calculate confidence difference
            model_conf = confidence_scores[i]
            conf_diff = model_conf - manual_confidence
            confidence_differences.append(conf_diff)

        results_df['manual_label_confidence'] = manual_label_confidences
        results_df['confidence_difference'] = confidence_differences
        
        # Flag potential errors
        potential_errors = []
        error_reasons = []
        
        for _, row in results_df.iterrows():
            is_error = False
            reasons = []
            
            # Criterion 1: Model disagrees and is highly confident
            if not row['prediction_matches'] and row['model_confidence'] >= self.confidence_threshold:
                is_error = True
                reasons.append(f"Model confident ({row['model_confidence']:.2f}) in different label")
            
            # Criterion 2: Large confidence difference
            if row['confidence_difference'] >= self.disagreement_threshold:
                is_error = True
                reasons.append(f"Model much more confident (+{row['confidence_difference']:.2f})")

            # Criterion 3: Very low confidence in manual label
            if row['manual_label_confidence'] < 0.1 and row['model_confidence'] > 0.6:
                is_error = True
                reasons.append(f"Very low confidence ({row['manual_label_confidence']:.2f}) in manual label")

            potential_errors.append(is_error)
            error_reasons.append('; '.join(reasons) if reasons else 'No issues detected')
        
        results_df['potential_error'] = potential_errors
        results_df['error_reason'] = error_reasons
        
        # Calculate statistics
        total_samples = len(results_df)
        matching_predictions = results_df['prediction_matches'].sum()
        potential_error_count = results_df['potential_error'].sum()
        accuracy = matching_predictions / total_samples
        
        print(f"Analysis completed:")
        print(f"   Total samples: {total_samples}")
        print(f"   Matching predictions: {matching_predictions} ({accuracy:.1%})")
        print(f"   Potential manual errors flagged: {potential_error_count} ({potential_error_count/total_samples:.1%})")

        return results_df
    
    def create_detailed_report(self, results_df, output_dir=None):
        """Create analysis reports."""
        print("\n" + "="*60)
        print("CREATING DETAILED ANALYSIS REPORTS")
        print("="*60)
        
        if output_dir is None:
            output_dir = os.path.join(self.config['paths']['processed_data'], 'prediction_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save complete analysis
        complete_report_path = os.path.join(output_dir, 'complete_prediction_analysis.csv')
        results_df.to_csv(complete_report_path, index=False)
        print(f"Complete analysis saved: {complete_report_path}")
        
        # 2. Save potential errors only
        error_df = results_df[results_df['potential_error']]
        error_report_path = os.path.join(output_dir, 'potential_manual_labeling_errors.csv')
        error_df.to_csv(error_report_path, index=False)
        print(f"Potential errors saved: {error_report_path}")
        
        # 3. Save correct predictions
        correct_df = results_df[results_df['prediction_matches']]
        correct_report_path = os.path.join(output_dir, 'correct_predictions.csv')
        correct_df.to_csv(correct_report_path, index=False)
        print(f"Correct predictions saved: {correct_report_path}")
        
        # 4. Create summary report
        self._create_summary_report(results_df, output_dir)
        
        # 5. Create detailed error analysis
        if len(error_df) > 0:
            self._create_error_analysis_report(error_df, output_dir)
        
        # 6. Create prediction accuracy by class
        self._create_class_accuracy_report(results_df, output_dir)
        
        print(f"All reports saved to: {output_dir}")
        return output_dir
    
    def _create_summary_report(self, results_df, output_dir):
        """Create summary statistics report."""
        summary_path = os.path.join(output_dir, 'prediction_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("AUTOMATIC PREDICTION ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall statistics
            total = len(results_df)
            correct = results_df['prediction_matches'].sum()
            errors_flagged = results_df['potential_error'].sum()
            accuracy = correct / total
            
            f.write(f"OVERALL PERFORMANCE:\n")
            f.write(f"Total samples analyzed: {total}\n")
            f.write(f"Model predictions matching labels: {correct} ({accuracy:.1%})\n")
            f.write(f"Model disagreements with labels: {total - correct} ({(1-accuracy):.1%})\n")
            f.write(f"Potential manual labeling errors flagged: {errors_flagged} ({errors_flagged/total:.1%})\n\n")
            
            # Confidence analysis
            f.write(f"CONFIDENCE ANALYSIS:\n")
            avg_model_conf = results_df['model_confidence'].mean()
            avg_manual_conf = results_df['manual_label_confidence'].mean()
            f.write(f"Average model confidence: {avg_model_conf:.2f}\n")
            f.write(f"Average confidence in manual labels: {avg_manual_conf:.2f}\n")
            f.write(f"Average confidence difference: {results_df['confidence_difference'].mean():.2f}\n\n")
            
            # Most common disagreements
            disagreements = results_df[~results_df['prediction_matches']]
            if len(disagreements) > 0:
                f.write(f"MOST COMMON DISAGREEMENTS (Manual → Model):\n")
                disagreement_pairs = disagreements.groupby(['problem_type', 'model_prediction']).size().sort_values(ascending=False)
                for (manual_label, model_pred), count in disagreement_pairs.head(10).items():
                    f.write(f"  {manual_label} → {model_pred}: {count} cases\n")
                f.write("\n")
            
            # Error reasons
            if errors_flagged > 0:
                f.write(f"ERROR FLAGGING REASONS:\n")
                error_reasons = results_df[results_df['potential_error']]['error_reason'].value_counts()
                for reason, count in error_reasons.head(10).items():
                    f.write(f"  {reason}: {count} cases\n")
            
            # Recommendations
            f.write(f"\nRECOMMENDATIONS:\n")
            if accuracy > 0.9:
                f.write("Excellent agreement (>90%) between model and manual labels\n")
            elif accuracy > 0.8:
                f.write("Good agreement (>80%) between model and manual labels\n")
            elif accuracy > 0.7:
                f.write("Moderate agreement (70-80%) - review disagreements\n")
            else:
                f.write("Low agreement (<70%) - significant review needed\n")

            if errors_flagged > 0:
                f.write(f"🔍 Review {errors_flagged} flagged potential errors manually\n")
            
        print(f"Summary report saved: {summary_path}")
    
    def _create_error_analysis_report(self, error_df, output_dir):
        """Create detailed error analysis for manual review."""
        error_analysis_path = os.path.join(output_dir, 'error_analysis_for_review.txt')
        
        with open(error_analysis_path, 'w') as f:
            f.write("POTENTIAL ERRORS - DETAILED ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total potential errors flagged: {len(error_df)}\n")
            f.write("Review each case below to determine if correction is needed.\n\n")
            
            for i, (_, row) in enumerate(error_df.iterrows(), 1):
                f.write(f"CASE {i}:\n")
                f.write(f"Description: {row['description']}\n")
                f.write(f"Human Label: {row['problem_type']}\n")
                f.write(f"Model Suggests: {row['model_prediction']}\n")
                f.write(f"Model Confidence: {row['model_confidence']:.2f}\n")
                f.write(f"Manual Label Confidence: {row['manual_label_confidence']:.2f}\n")
                f.write(f"Reason Flagged: {row['error_reason']}\n")
                f.write(f"Action: [ ] Keep Manual Label  [ ] Use Model Label  [ ] Different Label\n")
                f.write("-" * 60 + "\n\n")
        
        print(f"Error analysis for review saved: {error_analysis_path}")
    
    def _create_class_accuracy_report(self, results_df, output_dir):
        """Create per-class accuracy analysis."""
        class_accuracy_path = os.path.join(output_dir, 'class_accuracy_analysis.csv')
        
        # Calculate per-class statistics
        class_stats = []
        for class_name in results_df['problem_type'].unique():
            class_data = results_df[results_df['problem_type'] == class_name]
            
            total_samples = len(class_data)
            correct_predictions = class_data['prediction_matches'].sum()
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            avg_confidence = class_data['model_confidence'].mean()
            errors_flagged = class_data['potential_error'].sum()
            
            class_stats.append({
                'class_name': class_name,
                'total_samples': total_samples,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'average_model_confidence': avg_confidence,
                'potential_errors_flagged': errors_flagged,
                'error_rate': errors_flagged / total_samples if total_samples > 0 else 0
            })
        
        class_df = pd.DataFrame(class_stats)
        class_df = class_df.sort_values('accuracy', ascending=False)
        class_df.to_csv(class_accuracy_path, index=False)
        
        print(f"Class accuracy analysis saved: {class_accuracy_path}")
    
    def predict_and_analyze(self, csv_path, description_col='description', true_label_col='problem_type'):
        """
        Complete pipeline: load data, predict, analyze, and generate reports.
        
        Args:
            csv_path: Path to CSV file with descriptions and human labels
            description_col: Name of description column
            true_label_col: Name of true label column
        
        Returns:
            dict: Analysis results
        """
        print("AUTOMATIC PROBLEM TYPE PREDICTION AND ERROR CHECKING")
        print("="*70)
        
        try:
            # Load data
            print(f"Loading data from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Validate columns
            if description_col not in df.columns:
                raise ValueError(f"Description column '{description_col}' not found. Available: {list(df.columns)}")
            if true_label_col not in df.columns:
                raise ValueError(f"Label column '{true_label_col}' not found. Available: {list(df.columns)}")
            
            # Clean data
            df = df.dropna(subset=[description_col, true_label_col])
            print(f"Loaded {len(df)} samples for analysis")
            print(f"   Unique manual labels: {df[true_label_col].nunique()}")
            
            # Preprocess descriptions
            X_processed, cleaned_descriptions = self.preprocess_descriptions(df[description_col].values)
            
            # Make predictions
            predicted_labels, confidence_scores, all_predictions = self.predict_problem_types(X_processed)
            
            # Analyze results
            results_df = self.analyze_predictions(
                df, predicted_labels, confidence_scores, all_predictions, 
                description_col, true_label_col
            )
            
            # Create reports
            report_dir = self.create_detailed_report(results_df)
            
            # Return summary
            total_samples = len(results_df)
            matching_predictions = results_df['prediction_matches'].sum()
            potential_errors = results_df['potential_error'].sum()
            accuracy = matching_predictions / total_samples
            
            print(f"\n Analysis completed successfully!")
            print(f" Reports saved in: {report_dir}")
            
            return {
                'results_df': results_df,
                'report_directory': report_dir,
                'total_samples': total_samples,
                'matching_predictions': matching_predictions,
                'accuracy': accuracy,
                'potential_errors_flagged': potential_errors,
                'error_rate': potential_errors / total_samples
            }
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            return None
    

def main():
    """Main function."""
    print("Automatic Problem Type Predictor and Error Checker")
    print("="*60)
    print("Uses trained model to predict problem types and identify potential errors")
    print()
    
    # Get input file
    csv_path = input("Enter path to CSV file with problems (or press Enter for test data): ").strip()
    
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    # Get column names
    print("\nColumn configuration:")
    desc_col = input("Description column name: ")
    label_col = input("Problem type column name: ")
    
    # Run analysis
    predictor = ProblemTypePredictor()
    results = predictor.predict_and_analyze(csv_path, desc_col, label_col)
    
    if results:
        print(f"\nFINAL RESULTS:")
        print(f"   Total samples: {results['total_samples']}")
        print(f"   Model accuracy: {results['accuracy']:.1%}")
        print(f"   Potential errors flagged: {results['potential_errors_flagged']} ({results['error_rate']:.1%})")
        print(f"   Detailed reports: {results['report_directory']}")
    
    return predictor, results

if __name__ == "__main__":
    main()
