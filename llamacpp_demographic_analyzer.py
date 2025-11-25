#!/usr/bin/env python3
"""
Llama-cpp Hate Speech Analyzer with Demographic Perspective
Analyzes hate speech with different annotator demographic backgrounds
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import random
from typing import Optional, List, Dict
import json

# Import llama-cpp-python
try:
    from llama_cpp import Llama
except ImportError:
    print("❌ llama-cpp-python not installed!")
    print("Install with: pip install llama-cpp-python")
    sys.exit(1)

# Import huggingface_hub for model downloading
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("❌ huggingface_hub not installed!")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)

class DemographicHateAnalyzer:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the demographic-aware hate speech analyzer"""
        self.model_path = model_path
        self.llm = None
        self.model_info = {}
        
        # Predefined demographic profiles
        self.demographic_profiles = {
            "black_muslim_woman": {
                "description": "you are a black Muslim woman",
                "details": "You identify as a Black Muslim woman with lived experience of intersectional discrimination."
            },
            "white_christian_man": {
                "description": "you are a white Christian man", 
                "details": "You identify as a white Christian man from a traditional background."
            },
            "latina_atheist_woman": {
                "description": "you are a Latina atheist woman",
                "details": "You identify as a Latina atheist woman with progressive views."
            },
            "asian_buddhist_man": {
                "description": "you are an Asian Buddhist man",
                "details": "You identify as an Asian Buddhist man with moderate political views."
            },
            "mixed_race_agnostic_nonbinary": {
                "description": "you are a mixed-race agnostic non-binary person",
                "details": "You identify as a mixed-race agnostic non-binary person with diverse perspectives."
            },
            "neutral_baseline": {
                "description": "you are an objective annotator",
                "details": "You are an objective annotator without specified demographic characteristics."
            }
        }
        
    def download_gguf_model(self, model_id: str = "bartowski/Llama-3.2-3B-Instruct-GGUF", 
                           filename: str = "Llama-3.2-3B-Instruct-Q4_K_M.gguf") -> str:
        """Download GGUF model from Hugging Face"""
        print(f"🔽 Downloading {filename} from {model_id}...")
        
        try:
            model_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                cache_dir="./models"
            )
            
            print(f"✅ Model downloaded: {model_path}")
            self.model_info = {
                "repo_id": model_id,
                "filename": filename,
                "local_path": model_path,
                "size_gb": Path(model_path).stat().st_size / (1024**3)
            }
            
            return model_path
            
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            raise

    def initialize_llamacpp(self, model_path: str = None) -> bool:
        """Initialize llama.cpp with the GGUF model"""
        if model_path is None:
            model_path = self.model_path
            
        if model_path is None or not Path(model_path).exists():
            print("📥 No model found, downloading...")
            model_path = self.download_gguf_model()
        
        print(f"🚀 Initializing llama.cpp with {model_path}")
        
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            
            print("✅ Llama.cpp initialized successfully!")
            self.model_path = model_path
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize llama.cpp: {e}")
            return False

    def create_demographic_prompt(self, text: str, demographic_profile: str) -> str:
        """Create hate speech analysis prompt with demographic perspective"""
        
        if demographic_profile not in self.demographic_profiles:
            raise ValueError(f"Unknown demographic profile: {demographic_profile}")
        
        profile = self.demographic_profiles[demographic_profile]
        
        prompt = f"""As someone who {profile['description']}, analyze this comment for hate speech.

{profile['details']} Consider how this comment might affect you and people like you.

Hate speech definition: 'bias-motivated, hostile and malicious language targeted at a person/group because of their actual or perceived innate characteristics, especially when the group is unnecessarily labeled'

Comment: "{text}"

From your lived experience and perspective, rate this comment:
0 = No hate speech (not targeting or attacking any group)
1 = Unclear/borderline (potentially offensive but ambiguous) 
2 = Clear hate speech (clearly attacking or targeting a group)

Be decisive based on YOUR perspective. Respond with just the number (0, 1, or 2)."""

        return prompt

    def analyze_comment_with_demographics(self, text: str, demographic_profile: str) -> Dict:
        """Analyze a single comment with specified demographic perspective"""
        if self.llm is None:
            raise RuntimeError("Llama model not initialized!")
        
        prompt = self.create_demographic_prompt(text, demographic_profile)
        
        try:
            response = self.llm(
                prompt,
                max_tokens=200,
                temperature=0.1,
                stop=["Comment:", "Analyze:", "\n\n"]
            )
            
            response_text = response['choices'][0]['text'].strip()
            
            # Extract rating (first number found)
            rating = None
            for char in response_text:
                if char.isdigit():
                    rating = int(char)
                    break
            
            if rating is None or rating not in [0, 1, 2]:
                rating = 1  # Default to unclear if parsing fails
            
            # Categorize rating
            if rating == 0:
                category = "no_hate"
            elif rating == 1:
                category = "unclear" 
            else:
                category = "hate"
            
            return {
                "rating": rating,
                "category": category,
                "demographic_profile": demographic_profile,
                "explanation": response_text,
                "confidence": self._calculate_confidence(response_text),
                "method": "llama-cpp-gguf-demographic"
            }
            
        except Exception as e:
            print(f"⚠️  Error analyzing comment: {e}")
            return {
                "rating": 1,
                "category": "unclear",
                "demographic_profile": demographic_profile,
                "explanation": f"Error: {str(e)}",
                "confidence": 0.0,
                "method": "llama-cpp-gguf-demographic"
            }

    def _calculate_confidence(self, response_text: str) -> float:
        """Calculate confidence score based on response certainty"""
        confidence_indicators = {
            "clearly": 0.9,
            "definitely": 0.9,
            "obviously": 0.8,
            "likely": 0.7,
            "probably": 0.7,
            "seems": 0.6,
            "might": 0.4,
            "possibly": 0.4,
            "unclear": 0.3,
            "unsure": 0.2
        }
        
        text_lower = response_text.lower()
        max_confidence = 0.5  # Default confidence
        
        for indicator, conf in confidence_indicators.items():
            if indicator in text_lower:
                max_confidence = max(max_confidence, conf)
        
        return max_confidence

    def analyze_batch_with_demographics(self, comments: List[str], comment_ids: List[str], 
                                      demographic_profile: str, max_comments: int = 50) -> pd.DataFrame:
        """Analyze multiple comments with specified demographic perspective"""
        
        if len(comments) != len(comment_ids):
            raise ValueError("Comments and comment_ids must have same length")
        
        # Limit number of comments
        if len(comments) > max_comments:
            print(f"📊 Limiting analysis to {max_comments} comments (from {len(comments)} available)")
            indices = random.sample(range(len(comments)), max_comments)
            comments = [comments[i] for i in indices]
            comment_ids = [comment_ids[i] for i in indices]
        
        print(f"\n🤖 Analyzing {len(comments)} comments with demographic perspective: {demographic_profile}")
        print(f"👤 Profile: {self.demographic_profiles[demographic_profile]['description']}")
        print("=" * 60)
        
        results = []
        
        for i, (comment_id, text) in enumerate(zip(comment_ids, comments), 1):
            print(f"📝 [{i:2d}/{len(comments)}] Analyzing comment {comment_id}... ", end="", flush=True)
            
            result = self.analyze_comment_with_demographics(text, demographic_profile)
            result.update({
                "comment_id": comment_id,
                "text": text,
                "analysis_order": i
            })
            results.append(result)
            
            print(f"Rating: {result['rating']} ({result['category']})")
        
        df = pd.DataFrame(results)
        
        # Print summary
        print(f"\n📊 ANALYSIS SUMMARY ({demographic_profile})")
        print("=" * 40)
        print(f"Total comments: {len(df)}")
        print(f"Hate speech (2): {len(df[df['rating']==2])} ({len(df[df['rating']==2])/len(df)*100:.1f}%)")
        print(f"Unclear (1): {len(df[df['rating']==1])} ({len(df[df['rating']==1])/len(df)*100:.1f}%)")  
        print(f"No hate (0): {len(df[df['rating']==0])} ({len(df[df['rating']==0])/len(df)*100:.1f}%)")
        print(f"Avg confidence: {df['confidence'].mean():.3f}")
        
        return df

    def save_results(self, df: pd.DataFrame, demographic_profile: str) -> str:
        """Save analysis results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demographic_hate_analysis_{demographic_profile}_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        print(f"\n💾 Results saved: {filename}")
        
        return filename

def load_dataset(file_path: str = "processed_dataset.csv") -> pd.DataFrame:
    """Load the hate speech dataset"""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    print(f"📚 Loading dataset: {file_path}")
    df = pd.read_csv(file_path)
    print(f"📊 Loaded {len(df)} comments")
    
    return df

def select_random_comments(df: pd.DataFrame, n: int = 50) -> tuple:
    """Select random comments for analysis"""
    print(f"\n🎲 Selecting {n} random comments...")
    
    # Get unique comments (some comment_ids might be duplicated due to multiple annotators)
    df_unique = df.drop_duplicates('comment_id')
    
    if len(df_unique) < n:
        print(f"⚠️  Only {len(df_unique)} unique comments available, using all")
        selected = df_unique
    else:
        selected = df_unique.sample(n=n, random_state=42)
    
    comments = selected['text'].tolist()
    comment_ids = selected['comment_id'].tolist()
    
    print(f"✅ Selected {len(comments)} comments")
    return comments, comment_ids

def run_demographic_analysis(demographic_profile: str, num_comments: int = 50):
    """Run analysis with a specific demographic profile"""
    print("🎯 DEMOGRAPHIC-AWARE HATE SPEECH ANALYSIS")  
    print("🦙 Llama-cpp with Demographic Perspectives")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DemographicHateAnalyzer()
    
    print(f"\n🎯 Selected profile: {demographic_profile}")
    print(f"👤 Description: {analyzer.demographic_profiles[demographic_profile]['description']}")
    print(f"📝 Details: {analyzer.demographic_profiles[demographic_profile]['details']}")
    
    try:
        # Initialize model
        if not analyzer.initialize_llamacpp():
            return None
        
        # Load dataset
        df = load_dataset()
        
        # Select random comments
        comments, comment_ids = select_random_comments(df, n=num_comments)
        
        # Analyze with demographic perspective
        results_df = analyzer.analyze_batch_with_demographics(
            comments, comment_ids, demographic_profile, max_comments=num_comments
        )
        
        # Save results
        output_file = analyzer.save_results(results_df, demographic_profile)
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print("=" * 25)
        print(f"📊 Analyzed: {len(results_df)} comments")
        print(f"👤 Perspective: {demographic_profile}")
        print(f"💾 Output: {output_file}")
        print(f"🤖 Model: {analyzer.model_info.get('filename', 'Unknown')}")
        
        return results_df
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        raise

def main():
    """Main analysis function - easy to customize"""
    
    # Available demographic profiles
    analyzer = DemographicHateAnalyzer()
    
    print("👥 Available demographic profiles:")
    for key, profile in analyzer.demographic_profiles.items():
        print(f"   {key}: {profile['description']}")
    
    # ==== CUSTOMIZE HERE ====
    # Change this to any profile from the list above:
    demographic_profile = "white_christian_man"
    num_comments = 50
    # =======================
    
    # Run the analysis
    results = run_demographic_analysis(demographic_profile, num_comments)
    
    return results

if __name__ == "__main__":
    main()