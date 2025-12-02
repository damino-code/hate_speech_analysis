#!/usr/bin/env python3
"""
llama-cpp Multi-Attribute Comment Analyzer - Likert Scale Version
Uses text-based Likert responses (Strongly disagree, Disagree, etc.) instead of numeric values
Then converts responses to numeric scores based on attribute type (binary or multi-class)
"""

import pandas as pd
import json
from datetime import datetime
import time
import os
from pathlib import Path
import re

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

class LikertMultiAttributeAnalyzer:
    """Analyzer that uses Likert scale text responses"""
    
    def __init__(self, model):
        self.model = model
        
        # Define attribute configurations with their Likert scale mappings
        self.attribute_config = {
            'sentiment': {
                'type': 'multiclass',
                'scale': 5,
                'question': 'The comment expresses positive sentiment',
                'options': ['Strongly disagree', 'Disagree', 'Neither disagree nor agree', 'Agree', 'Strongly agree'],
                'mapping': {'strongly disagree': 1.0, 'disagree': 2.0, 'neither disagree nor agree': 3.0, 'agree': 4.0, 'strongly agree': 5.0}
            },
            'respect': {
                'type': 'binary',
                'scale': 2,
                'question': 'The comment is respectful',
                'options': ['Disagree', 'Agree'],
                'mapping': {'disagree': 1.0, 'agree': 2.0}
            },
            'insult': {
                'type': 'multiclass',
                'scale': 4,
                'question': 'The comment contains insults',
                'options': ['Strongly disagree', 'Disagree', 'Agree', 'Strongly agree'],
                'mapping': {'strongly disagree': 1.0, 'disagree': 2.0, 'agree': 3.0, 'strongly agree': 4.0}
            },
            'humiliate': {
                'type': 'multiclass',
                'scale': 3,
                'question': 'The comment attempts to humiliate',
                'options': ['Disagree', 'Neither disagree nor agree', 'Agree'],
                'mapping': {'disagree': 1.0, 'neither disagree nor agree': 2.0, 'agree': 3.0}
            },
            'status': {
                'type': 'binary',
                'scale': 2,
                'question': 'The comment treats the target as equal or superior',
                'options': ['Disagree', 'Agree'],
                'mapping': {'disagree': 1.0, 'agree': 2.0}
            },
            'dehumanize': {
                'type': 'binary',
                'scale': 2,
                'question': 'The comment dehumanizes the target',
                'options': ['Disagree', 'Agree'],
                'mapping': {'disagree': 1.0, 'agree': 2.0}
            },
            'violence': {
                'type': 'binary',
                'scale': 2,
                'question': 'The comment calls for violence',
                'options': ['Disagree', 'Agree'],
                'mapping': {'disagree': 1.0, 'agree': 2.0}
            },
            'genocide': {
                'type': 'binary',
                'scale': 2,
                'question': 'The comment calls for genocide',
                'options': ['Disagree', 'Agree'],
                'mapping': {'disagree': 1.0, 'agree': 2.0}
            },
            'attack_defend': {
                'type': 'multiclass',
                'scale': 4,
                'question': 'The comment is attacking (rather than defending)',
                'options': ['Strongly disagree', 'Disagree', 'Agree', 'Strongly agree'],
                'mapping': {'strongly disagree': 1.0, 'disagree': 2.0, 'agree': 3.0, 'strongly agree': 4.0}
            }
        }
        
        # Build prompt template
        self.prompt_template = self._build_prompt_template()
    
    def _build_prompt_template(self):
        """Build comprehensive prompt template with all attributes"""
        prompt = '''Analyze the following comment using Likert scale responses.

Comment: "{text}"

For each statement, respond with one of the provided options (EXACTLY as written):

'''
        
        for i, (attr, config) in enumerate(self.attribute_config.items(), 1):
            prompt += f"{i}. {config['question']}\n"
            prompt += f"   Options: {', '.join(config['options'])}\n"
            prompt += f"   Response: \n\n"
        
        prompt += '''IMPORTANT: 
- Respond with ONLY the option text from the provided choices
- Use the exact wording provided (case-insensitive is ok)
- One response per statement
- Format each response on a new line

Example format:
1. Disagree
2. Strongly agree
3. Neither disagree nor agree
etc.

Your responses:
'''
        
        return prompt
    
    def analyze_with_gguf(self, text):
        """Analyze text using Likert scale responses"""
        try:
            prompt = self.prompt_template.format(text=text[:400])
            
            response = self.model(prompt, max_tokens=150, temperature=0.1)
            output = response['choices'][0]['text'].strip()
            
            # Parse Likert responses
            likert_responses = self._parse_likert_responses(output)
            
            # Convert to numeric values
            numeric_results = self._convert_to_numeric(likert_responses)
            
            # Add metadata
            numeric_results.update({
                'raw_output': output[:200],
                'text': text[:100],
                'likert_responses': likert_responses,
                'prompt_type': 'likert'
            })
            
            return numeric_results
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return self._fallback_analysis(text)
    
    def _parse_likert_responses(self, output):
        """Parse Likert scale text responses from LLM output"""
        responses = {}
        output_lower = output.lower()
        
        # Try to extract responses for each attribute
        for attr, config in self.attribute_config.items():
            found = False
            
            # Look for any of the valid options in the output
            for option in config['options']:
                option_lower = option.lower()
                # Search for the option as a complete phrase
                pattern = r'\b' + re.escape(option_lower) + r'\b'
                if re.search(pattern, output_lower):
                    responses[attr] = option.lower()
                    found = True
                    break
            
            if not found:
                # Default to neutral/middle option if available
                if len(config['options']) % 2 == 1:  # Odd number of options
                    middle_idx = len(config['options']) // 2
                    responses[attr] = config['options'][middle_idx].lower()
                else:  # Even number of options - choose middle-lower
                    middle_idx = len(config['options']) // 2 - 1
                    responses[attr] = config['options'][middle_idx].lower()
        
        return responses
    
    def _convert_to_numeric(self, likert_responses):
        """Convert Likert text responses to numeric values"""
        numeric_results = {
            'sentiment': 3.0,
            'respect': 1.5,
            'insult': 1.0,
            'humiliate': 1.0,
            'status': 1.5,
            'dehumanize': 1.0,
            'violence': 1.0,
            'genocide': 1.0,
            'attack_defend': 2.0,
            'confidence': 0.7
        }
        
        conversion_success = 0
        
        for attr, likert_text in likert_responses.items():
            if attr in self.attribute_config:
                config = self.attribute_config[attr]
                mapping = config['mapping']
                
                # Try to match the response to a valid option
                if likert_text in mapping:
                    numeric_results[attr] = mapping[likert_text]
                    conversion_success += 1
                else:
                    # Try partial matching
                    for option_text, value in mapping.items():
                        if option_text in likert_text or likert_text in option_text:
                            numeric_results[attr] = value
                            conversion_success += 1
                            break
        
        # Adjust confidence based on conversion success rate
        numeric_results['confidence'] = conversion_success / len(self.attribute_config)
        
        return numeric_results
    
    def _fallback_analysis(self, text):
        """Fallback content-based analysis when parsing fails"""
        text_lower = text.lower()
        
        results = {
            'sentiment': 3.0,
            'respect': 2.0,
            'insult': 1.0,
            'humiliate': 1.0,
            'status': 2.0,
            'dehumanize': 1.0,
            'violence': 1.0,
            'genocide': 1.0,
            'attack_defend': 2.0,
            'confidence': 0.5,
            'raw_output': 'fallback_analysis',
            'text': text[:100],
            'likert_responses': {},
            'prompt_type': 'likert'
        }
        
        # Simple keyword-based analysis
        hate_words = ['hate', 'kill', 'die', 'murder', 'eliminate', 'destroy']
        positive_words = ['love', 'respect', 'appreciate', 'admire', 'support']
        
        if any(word in text_lower for word in hate_words):
            results['sentiment'] = 1.0
            results['respect'] = 1.0
            results['insult'] = 4.0
            results['attack_defend'] = 4.0
            
        if any(word in text_lower for word in positive_words):
            results['sentiment'] = 5.0
            results['respect'] = 2.0
            results['attack_defend'] = 1.0
        
        return results

def analyze_comments(csv_file, output_file=None, sample_size=200):
    """Analyze comments using Likert scale prompting"""
    print("=" * 60)
    print("🎯 LIKERT SCALE MULTI-ATTRIBUTE ANALYSIS")
    print("=" * 60)
    
    # Load model
    model = load_llama_model()
    if not model:
        return
    
    analyzer = LikertMultiAttributeAnalyzer(model)
    
    # Load comments
    print(f"\n📄 Loading comments from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Determine text column
    text_columns = ['text', 'comment', 'content', 'message']
    text_column = None
    for col in text_columns:
        if col in df.columns:
            text_column = col
            break
    
    if not text_column:
        print(f"❌ No text column found. Available: {list(df.columns)}")
        return
    
    print(f"📝 Using text column: '{text_column}'")
    
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
        comment_text = str(row.get(text_column, ''))
        if len(comment_text.strip()) == 0:
            continue
            
        comment_id = row.get('comment_id', idx)
        
        print(f"\n[{len(results)+1}/{len(df_sample)}] Analyzing comment {comment_id}...")
        print(f"Text preview: {comment_text[:80]}...")
        
        result = analyzer.analyze_with_gguf(comment_text)
        result['comment_id'] = comment_id
        result['index'] = idx
        
        results.append(result)
        
        # Show Likert responses and converted values
        likert_resp = result.get('likert_responses', {})
        if likert_resp:
            print(f"✓ Likert responses:")
            print(f"  Sentiment: {likert_resp.get('sentiment', 'N/A')} → {result['sentiment']:.1f}")
            print(f"  Respect: {likert_resp.get('respect', 'N/A')} → {result['respect']:.1f}")
            print(f"  Insult: {likert_resp.get('insult', 'N/A')} → {result['insult']:.1f}")
            print(f"  Confidence: {result['confidence']:.2f}")
        else:
            print(f"✓ Values: Sent={result['sentiment']:.1f}, Resp={result['respect']:.1f}, Ins={result['insult']:.1f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"likert_analysis_{timestamp}.csv"
    
    results_df.to_csv(output_file, index=False)
    
    # Also save Likert responses separately for analysis
    likert_only = []
    for r in results:
        likert_entry = {
            'comment_id': r['comment_id'],
            'text': r['text']
        }
        if 'likert_responses' in r:
            for attr, response in r['likert_responses'].items():
                likert_entry[f'{attr}_likert'] = response
                likert_entry[f'{attr}_numeric'] = r[attr]
        likert_only.append(likert_entry)
    
    likert_df = pd.DataFrame(likert_only)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    likert_file = f"likert_responses_{timestamp}.csv"
    likert_df.to_csv(likert_file, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Analysis complete!")
    print(f"📁 Numeric results saved: {output_file}")
    print(f"📁 Likert responses saved: {likert_file}")
    print(f"⏱️  Time: {elapsed:.1f}s ({elapsed/len(results):.1f}s per comment)")
    print(f"📊 Analyzed: {len(results)} comments")
    print(f"{'='*60}")
    
    # Print summary
    print(f"\n📈 SUMMARY:")
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    avg_sentiment = sum(r['sentiment'] for r in results) / len(results)
    print(f"  Average confidence: {avg_confidence:.2f}")
    print(f"  Average sentiment: {avg_sentiment:.2f}")
    
    # Count successful Likert parsing
    successful_parses = sum(1 for r in results if r.get('likert_responses', {}))
    print(f"  Successful Likert parses: {successful_parses}/{len(results)} ({successful_parses/len(results)*100:.1f}%)")

if __name__ == "__main__":
    import sys
    
    csv_file = "selected_comments.csv"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        print("Usage: python llamacpp_likert_analyzer.py [input_csv]")
        sys.exit(1)
    
    analyze_comments(csv_file)
