#!/usr/bin/env python3
"""
llama-cpp Multi-Attribute Comment Analyzer - Vanilla Prompting
Pure analysis without any annotator demographic information
Uses simple, neutral prompts to analyze hate speech attributes
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

class VanillaMultiAttributeAnalyzer:
    """Vanilla analyzer without any annotator demographic information"""
    
    def __init__(self, model):
        self.model = model
        
        # Simple, neutral prompt template
        self.prompt_template = """Analyze the following comment and rate it on multiple attributes.

Comment: "{text}"

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
    
    def analyze_with_gguf(self, text):
        """Analyze text with vanilla prompting"""
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
                'prompt_type': 'vanilla'
            }
            
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
            return {
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
                'prompt_type': 'vanilla',
                'raw_output': '',
                'text': text[:100]
            }

def analyze_comments(csv_file, output_file=None, sample_size=200):
    """Analyze comments using vanilla prompting"""
    print("=" * 60)
    print("🎯 VANILLA MULTI-ATTRIBUTE ANALYSIS")
    print("=" * 60)
    
    # Load model
    model = load_llama_model()
    if not model:
        return
    
    analyzer = VanillaMultiAttributeAnalyzer(model)
    
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
    
    # Analyze each comment
    results = []
    start_time = time.time()
    
    for idx, row in df_sample.iterrows():
        comment_text = str(row.get('comment', ''))
        comment_id = row.get('comment_id', idx)
        
        print(f"\n[{len(results)+1}/{len(df_sample)}] Analyzing comment {comment_id}...")
        print(f"Text preview: {comment_text[:80]}...")
        
        result = analyzer.analyze_with_gguf(comment_text)
        result['comment_id'] = comment_id
        result['index'] = idx
        
        results.append(result)
        
        print(f"✓ Sentiment: {result['sentiment']:.1f}, Respect: {result['respect']:.1f}, "
              f"Insult: {result['insult']:.1f}, Confidence: {result['confidence']:.2f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"vanilla_analysis_{timestamp}.csv"
    
    results_df.to_csv(output_file, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Analysis complete!")
    print(f"📁 Results saved: {output_file}")
    print(f"⏱️  Time: {elapsed:.1f}s ({elapsed/len(results):.1f}s per comment)")
    print(f"📊 Analyzed: {len(results)} comments")
    print(f"{'='*60}")

if __name__ == "__main__":
    import sys
    
    csv_file = "selected_comments.csv"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        print("Usage: python llamacpp_vanilla_analyzer.py [input_csv]")
        sys.exit(1)
    
    analyze_comments(csv_file)
