#!/usr/bin/env python3
"""
llama-cpp Multi-Attribute Comment Analyzer - Annotator Role Prompting
Analyzes comments by simulating different annotator demographic perspectives
Tests if specifying annotator characteristics (gender, demographics, religion) affects ratings
"""

import pandas as pd
import json
from datetime import datetime
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel, Field
from typing import Optional
import re

class AnalysisAttributes(BaseModel):
    """Pydantic schema for multi-attribute comment analysis"""
    
    sentiment: float = Field(..., ge=1.0, le=5.0)
    respect: float = Field(..., ge=1.0, le=2.0)
    insult: float = Field(..., ge=1.0, le=4.0)
    humiliate: float = Field(..., ge=1.0, le=3.0)
    status: float = Field(..., ge=1.0, le=2.0)
    dehumanize: float = Field(..., ge=1.0, le=2.0)
    violence: float = Field(..., ge=1.0, le=2.0)
    genocide: float = Field(..., ge=1.0, le=2.0)
    attack_defend: float = Field(..., ge=1.0, le=4.0)
    confidence: float = Field(..., ge=0.1, le=1.0)

def find_gguf_models():
    """Find available GGUF model files"""
    print("🔍 Looking for GGUF files...")
    gguf_paths = []
    search_dirs = ["./gguf_models/", "~/.cache/huggingface/hub/", "./models/", "./"]
    
    for search_dir in search_dirs:
        path = Path(search_dir).expanduser()
        if path.exists():
            gguf_paths.extend(path.glob("*.gguf"))
            gguf_paths.extend(path.glob("**/*.gguf"))
    
    if gguf_paths:
        print(f"✅ Found {len(gguf_paths)} GGUF files")
        for i, gguf_path in enumerate(gguf_paths, 1):
            size_gb = gguf_path.stat().st_size / (1024**3)
            print(f"  {i}. {gguf_path.name} ({size_gb:.1f} GB)")
        return gguf_paths
    else:
        print("❌ No GGUF files found")
        return []

def load_llama_model(model_path=None):
    """Load llama-cpp model"""
    try:
        from llama_cpp import Llama
        
        if not model_path:
            gguf_paths = find_gguf_models()
            if not gguf_paths:
                raise FileNotFoundError("No GGUF models found")
            model_path = str(gguf_paths[0])
        
        print(f"\n🚀 Loading model...")
        print(f"📁 Path: {model_path}")
        
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0
        )
        
        print("✅ Model loaded successfully!")
        return llm
        
    except ImportError:
        print("❌ Error: llama-cpp-python not installed")
        print("Install with: pip install llama-cpp-python")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

class AnnotatorRoleAnalyzer:
    """Analyzer that prompts with specific annotator demographic roles"""
    
    def __init__(self, model, annotator_profile):
        self.model = model
        self.annotator_profile = annotator_profile
        
        # Build annotator description
        role_desc = self._build_role_description(annotator_profile)
        
        # Prompt template with annotator role
        self.prompt_template = f"""You are analyzing online comments as a content moderator with the following background:
{role_desc}

Based on your perspective and experience, analyze the following comment and rate it on multiple attributes.

Comment: "{{text}}"

Provide ratings for these attributes (output only numbers separated by spaces):
1. Sentiment (1=very negative, 2=negative, 3=neutral, 4=positive, 5=very positive)
2. Respect (1=not respectful, 2=respectful)
3. Insult (1=not insulting, 2=somewhat insulting, 3=insulting, 4=very insulting)
4. Humiliate (1=not humiliating, 2=neutral, 3=humiliating)
5. Status (1=implies inferior status, 2=equal or superior status)
6. Dehumanize (1=not dehumanizing, 2=dehumanizing)
7. Violence (1=no violence, 2=calls for violence)
8. Genocide (1=no genocide, 2=calls for genocide)
9. Attack/Defend (1=strongly defending, 2=somewhat defending, 3=somewhat attacking, 4=strongly attacking)
10. Confidence (0.1 to 1.0, how confident are you in this analysis?)

Output format: [sentiment] [respect] [insult] [humiliate] [status] [dehumanize] [violence] [genocide] [attack_defend] [confidence]

Ratings: """
    
    def _build_role_description(self, profile):
        """Build annotator role description from profile"""
        parts = []
        
        if 'gender' in profile:
            parts.append(f"Gender: {profile['gender']}")
        
        if 'age' in profile:
            parts.append(f"Age: {profile['age']}")
        
        if 'race' in profile:
            parts.append(f"Race/Ethnicity: {profile['race']}")
        
        if 'religion' in profile:
            parts.append(f"Religion: {profile['religion']}")
        
        if 'education' in profile:
            parts.append(f"Education: {profile['education']}")
        
        if 'ideology' in profile:
            parts.append(f"Political ideology: {profile['ideology']}")
        
        if 'income' in profile:
            parts.append(f"Income level: {profile['income']}")
        
        return "\n".join(parts) if parts else "General annotator"
    
    def analyze_with_gguf(self, text):
        """Analyze text with annotator role prompting"""
        try:
            prompt = self.prompt_template.format(text=text[:400])
            response = self.model(prompt, max_tokens=120, temperature=0.1)
            output = response['choices'][0]['text'].strip().lower()
            
            # Default values
            results = {
                'sentiment': 3.0,
                'respect': 1.5,
                'insult': 1.0,
                'humiliate': 1.0,
                'status': 1.5,
                'dehumanize': 1.0,
                'violence': 1.0,
                'genocide': 1.0,
                'attack_defend': 2.0,
                'confidence': 0.7,
                'prompt_type': 'annotator_role'
            }
            
            # Add annotator profile info to results
            for key, value in self.annotator_profile.items():
                results[f'annotator_{key}'] = value
            
            # Extract numbers from output
            number_sequence = re.findall(r'([0-9]\.?[0-9]*)', output)
            
            if len(number_sequence) >= 9:
                try:
                    sentiment = float(number_sequence[0])
                    respect = float(number_sequence[1])
                    insult = float(number_sequence[2])
                    humiliate = float(number_sequence[3])
                    status = float(number_sequence[4])
                    dehumanize = float(number_sequence[5])
                    violence = float(number_sequence[6])
                    genocide = float(number_sequence[7])
                    attack_defend = float(number_sequence[8])
                    confidence = float(number_sequence[9]) if len(number_sequence) > 9 else 0.7
                    
                    # Validate and assign
                    if 1.0 <= sentiment <= 5.0:
                        results['sentiment'] = sentiment
                    if 1.0 <= respect <= 2.0:
                        results['respect'] = respect
                    if 1.0 <= insult <= 4.0:
                        results['insult'] = insult
                    if 1.0 <= humiliate <= 3.0:
                        results['humiliate'] = humiliate
                    if 1.0 <= status <= 2.0:
                        results['status'] = status
                    if 1.0 <= dehumanize <= 2.0:
                        results['dehumanize'] = dehumanize
                    if 1.0 <= violence <= 2.0:
                        results['violence'] = violence
                    if 1.0 <= genocide <= 2.0:
                        results['genocide'] = genocide
                    if 1.0 <= attack_defend <= 4.0:
                        results['attack_defend'] = attack_defend
                    if 0.1 <= confidence <= 1.0:
                        results['confidence'] = confidence
                        
                except (ValueError, IndexError):
                    pass
            
            # Fallback: keyword-based patterns
            else:
                patterns = {
                    'sentiment': r'sentiment\s*[:=]?\s*([1-5]\.?[0-9]*)',
                    'respect': r'respect\s*[:=]?\s*([1-2]\.?[0-9]*)',
                    'insult': r'insult\s*[:=]?\s*([1-4]\.?[0-9]*)',
                    'humiliate': r'humiliate\s*[:=]?\s*([1-3]\.?[0-9]*)',
                    'status': r'status\s*[:=]?\s*([1-2]\.?[0-9]*)',
                    'dehumanize': r'dehumanize\s*[:=]?\s*([1-2]\.?[0-9]*)',
                    'violence': r'violence\s*[:=]?\s*([1-2]\.?[0-9]*)',
                    'genocide': r'genocide\s*[:=]?\s*([1-2]\.?[0-9]*)',
                    'attack_defend': r'attack[_-]?defend\s*[:=]?\s*([1-4]\.?[0-9]*)',
                    'confidence': r'confidence\s*[:=]?\s*([0-1]\.?[0-9]*)'
                }
                
                for attr, pattern in patterns.items():
                    match = re.search(pattern, output, re.IGNORECASE)
                    if match:
                        try:
                            value = float(match.group(1))
                            results[attr] = value
                        except ValueError:
                            pass
            
            results['raw_output'] = output[:200]
            results['text'] = text[:100]
            
            return results
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            base_results = {
                'sentiment': 3.0,
                'respect': 1.5,
                'insult': 1.0,
                'humiliate': 1.0,
                'status': 1.5,
                'dehumanize': 1.0,
                'violence': 1.0,
                'genocide': 1.0,
                'attack_defend': 2.0,
                'confidence': 0.5,
                'prompt_type': 'annotator_role',
                'raw_output': '',
                'text': text[:100]
            }
            
            # Add annotator profile
            for key, value in self.annotator_profile.items():
                base_results[f'annotator_{key}'] = value
            
            return base_results

def get_annotator_profiles():
    """Define different annotator profiles to test"""
    profiles = [
        # Gender-focused profiles
        {
            'profile_id': 'male_basic',
            'gender': 'male',
            'description': 'Male annotator'
        },
        {
            'profile_id': 'female_basic',
            'gender': 'female',
            'description': 'Female annotator'
        },
        
        # Demographic-focused profiles
        {
            'profile_id': 'young_white_male',
            'gender': 'male',
            'age': '25 years old',
            'race': 'White/Caucasian',
            'education': 'College degree',
            'description': 'Young white male with college education'
        },
        {
            'profile_id': 'middle_black_female',
            'gender': 'female',
            'age': '45 years old',
            'race': 'Black/African American',
            'education': 'Graduate degree',
            'description': 'Middle-aged Black female with graduate degree'
        },
        {
            'profile_id': 'older_asian_male',
            'gender': 'male',
            'age': '60 years old',
            'race': 'Asian',
            'education': 'High school',
            'description': 'Older Asian male with high school education'
        },
        
        # Religion-focused profiles
        {
            'profile_id': 'christian_conservative',
            'gender': 'male',
            'religion': 'Christian',
            'ideology': 'Conservative',
            'description': 'Christian male with conservative ideology'
        },
        {
            'profile_id': 'muslim_moderate',
            'gender': 'female',
            'religion': 'Muslim',
            'ideology': 'Moderate',
            'description': 'Muslim female with moderate ideology'
        },
        {
            'profile_id': 'jewish_liberal',
            'gender': 'male',
            'religion': 'Jewish',
            'ideology': 'Liberal',
            'description': 'Jewish male with liberal ideology'
        },
        {
            'profile_id': 'atheist_liberal',
            'gender': 'female',
            'religion': 'Atheist/Non-religious',
            'ideology': 'Liberal',
            'description': 'Atheist female with liberal ideology'
        },
        
        # Comprehensive profile
        {
            'profile_id': 'comprehensive_diverse',
            'gender': 'female',
            'age': '35 years old',
            'race': 'Hispanic/Latino',
            'religion': 'Catholic',
            'education': 'Graduate degree',
            'ideology': 'Moderate',
            'income': 'Middle income',
            'description': 'Hispanic Catholic female, 35, graduate degree, moderate ideology'
        }
    ]
    
    return profiles

def analyze_comments_with_roles(csv_file, output_prefix="annotator_role", sample_size=50):
    """Analyze comments with different annotator role perspectives"""
    print("=" * 60)
    print("🎯 ANNOTATOR ROLE MULTI-ATTRIBUTE ANALYSIS")
    print("=" * 60)
    
    # Load model
    model = load_llama_model()
    if not model:
        return
    
    # Load comments
    print(f"\n📄 Loading comments from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Sample comments
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"📊 Analyzing random sample of {sample_size} comments")
    else:
        df_sample = df
        print(f"📊 Analyzing all {len(df)} comments")
    
    # Get annotator profiles
    profiles = get_annotator_profiles()
    print(f"\n👥 Testing {len(profiles)} different annotator profiles:")
    for profile in profiles:
        print(f"  • {profile['profile_id']}: {profile['description']}")
    
    # Analyze each comment with each profile - save separate file per profile
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    total_analyses = len(df_sample) * len(profiles)
    current_analysis = 0
    
    saved_files = []
    all_results_combined = []  # For overall summary
    
    for profile in profiles:
        print(f"\n{'='*60}")
        print(f"👤 ANALYZING WITH PROFILE: {profile['profile_id']}")
        print(f"   {profile['description']}")
        print(f"{'='*60}")
        
        analyzer = AnnotatorRoleAnalyzer(model, profile)
        profile_results = []
        
        for idx, row in df_sample.iterrows():
            current_analysis += 1
            comment_text = str(row.get('comment', ''))
            comment_id = row.get('comment_id', idx)
            
            print(f"\n[{current_analysis}/{total_analyses}] Comment {comment_id} with {profile['profile_id']}...")
            print(f"Text: {comment_text[:80]}...")
            
            result = analyzer.analyze_with_gguf(comment_text)
            result['comment_id'] = comment_id
            result['index'] = idx
            result['profile_id'] = profile['profile_id']
            
            profile_results.append(result)
            all_results_combined.append(result)
            
            print(f"✓ Sent: {result['sentiment']:.1f}, Resp: {result['respect']:.1f}, "
                  f"Insult: {result['insult']:.1f}, Conf: {result['confidence']:.2f}")
        
        # Save individual profile results to separate CSV
        profile_df = pd.DataFrame(profile_results)
        profile_filename = f"{output_prefix}_{profile['profile_id']}_{timestamp}.csv"
        profile_df.to_csv(profile_filename, index=False)
        saved_files.append(profile_filename)
        
        print(f"\n💾 Saved {len(profile_results)} results for {profile['profile_id']}: {profile_filename}")
    
    # Also save combined file for convenience
    combined_df = pd.DataFrame(all_results_combined)
    combined_file = f"{output_prefix}_combined_{timestamp}.csv"
    combined_df.to_csv(combined_file, index=False)
    saved_files.append(combined_file)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Analysis complete!")
    print(f"📁 Individual profile files saved ({len(profiles)} files):")
    for f in saved_files[:-1]:  # All except combined
        print(f"   • {f}")
    print(f"📁 Combined file saved: {combined_file}")
    print(f"⏱️  Time: {elapsed:.1f}s ({elapsed/len(all_results_combined):.1f}s per analysis)")
    print(f"📊 Total analyses: {len(all_results_combined)} ({len(df_sample)} comments × {len(profiles)} profiles)")
    print(f"{'='*60}")
    
    # Print summary statistics
    print(f"\n📈 SUMMARY BY PROFILE:")
    print("-" * 60)
    for profile_id in combined_df['profile_id'].unique():
        profile_data = combined_df[combined_df['profile_id'] == profile_id]
        print(f"\n{profile_id}:")
        print(f"  Avg Sentiment: {profile_data['sentiment'].mean():.2f}")
        print(f"  Avg Insult: {profile_data['insult'].mean():.2f}")
        print(f"  Avg Violence: {profile_data['violence'].mean():.2f}")
        print(f"  Avg Confidence: {profile_data['confidence'].mean():.2f}")

if __name__ == "__main__":
    import sys
    
    csv_file = "selected_comments.csv"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        print("Usage: python llamacpp_annotator_role_analyzer.py [input_csv]")
        sys.exit(1)
    
    # Use smaller sample for role testing (50 comments × 10 profiles = 500 analyses)
    analyze_comments_with_roles(csv_file, sample_size=50)
