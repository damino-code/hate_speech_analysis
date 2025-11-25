#!/usr/bin/env python3
"""
llama-cpp Multi-Attribute Comment Analyzer (Refactored)
Analyzes sentiment and multiple attributes related to hate speech, respect, insults, etc.
Uses structured JSON output with Pydantic validation for robust analysis
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
import numpy as np

class AnalysisAttributes(BaseModel):
    """Pydantic schema for multi-attribute comment analysis with range validation"""
    
    sentiment: float = Field(..., ge=1.0, le=5.0, description="Sentiment rating from 1.0 (very negative) to 5.0 (very positive)")
    respect: float = Field(..., ge=1.0, le=2.0, description="Respect rating: 1.0 (not respectful) to 2.0 (respectful)")
    insult: float = Field(..., ge=1.0, le=4.0, description="Insult rating from 1.0 (not insulting) to 4.0 (very insulting)")
    humiliate: float = Field(..., ge=1.0, le=3.0, description="Humiliation rating from 1.0 (not humiliating) to 3.0 (humiliating)")
    status: float = Field(..., ge=1.0, le=2.0, description="Status rating: 1.0 (inferior status) to 2.0 (equal/superior status)")
    dehumanize: float = Field(..., ge=1.0, le=2.0, description="Dehumanization rating: 1.0 (not dehumanizing) to 2.0 (dehumanizing)")
    violence: float = Field(..., ge=1.0, le=2.0, description="Violence rating: 1.0 (no violence) to 2.0 (calls for violence)")
    genocide: float = Field(..., ge=1.0, le=2.0, description="Genocide rating: 1.0 (no genocide) to 2.0 (calls for genocide)")
    attack_defend: float = Field(..., ge=1.0, le=4.0, description="Attack-defend rating from 1.0 (defending) to 4.0 (attacking)")
    confidence: float = Field(..., ge=0.1, le=1.0, description="Confidence in analysis from 0.1 to 1.0")

def find_gguf_models():
    """Find available GGUF model files"""
    print("🔍 Looking for GGUF files...")
    gguf_paths = []
    search_dirs = ["./gguf_models/", "~/.cache/huggingface/hub/", "./models/", "./"]
    
    for search_dir in search_dirs:
        path = Path(search_dir).expanduser()
        if path.exists():
            gguf_paths.extend(path.glob("*.gguf"))
            # Also search in subdirectories for HF downloads
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

def initialize_llamacpp():
    """Initialize llama-cpp with available models"""
    print("🚀 INITIALIZING PURE LLAMA-CPP")
    print("=" * 40)
    
    try:
        from llama_cpp import Llama
        
        # Option 1: Look for existing GGUF files
        print("Option 1: Looking for existing GGUF models")
        gguf_paths = find_gguf_models()
        
        if gguf_paths:
            # Use the first GGUF file found
            model_path = str(gguf_paths[0])
            print(f"🦙 Loading: {gguf_paths[0].name}")
            
            # Load model with optimization for sentiment analysis
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,        # Context window
                n_batch=512,       # Batch size
                verbose=False,     # Reduce output
                n_threads=4,       # CPU threads
            )
            
            print(f"✅ Model loaded successfully")
            print(f"📁 Path: {model_path}")
            return MultiAttributeAnalyzer(llm)
        else:
            print("❌ No GGUF models found locally")
            print("Please download a GGUF model to use this analyzer.")
            return None
            
    except ImportError:
        print("❌ llama-cpp-python not installed")
        print("💡 Install with: pip install llama-cpp-python")
        return None
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return None

class MultiAttributeAnalyzer:
    def __init__(self, model):
        self.model = model
        
        if not self.model:
            raise Exception("No GGUF models available. Please download GGUF models.")

    def analyze_with_gguf(self, text):
        """Use direct GGUF model with structured JSON output for multi-attribute analysis"""
        try:
            # Create detailed prompt requesting JSON output
            prompt = f"""Analyze this comment and provide ratings as JSON:
"{text[:400]}"

Rate on these scales and output ONLY a JSON object:
{{"sentiment": 3.0, "respect": 2.0, "insult": 1.0, "humiliate": 1.0, "status": 2.0, "dehumanize": 1.0, "violence": 1.0, "genocide": 1.0, "attack_defend": 2.0, "confidence": 0.8}}

Scales:
- sentiment: 1.0 (very negative) to 5.0 (very positive)
- respect: 1.0 (not respectful) to 2.0 (respectful)  
- insult: 1.0 (not insulting) to 4.0 (very insulting)
- humiliate: 1.0 (not humiliating) to 3.0 (humiliating)
- status: 1.0 (inferior status) to 2.0 (equal/superior status)
- dehumanize: 1.0 (not dehumanizing) to 2.0 (dehumanizing)
- violence: 1.0 (no violence) to 2.0 (calls for violence)
- genocide: 1.0 (no genocide) to 2.0 (calls for genocide)
- attack_defend: 1.0 (defending) to 4.0 (attacking)
- confidence: 0.1 to 1.0 (confidence in analysis)

JSON only:"""
            
            # Simple call without grammar for compatibility
            response = self.model(prompt, max_tokens=200, temperature=0.1, stop=["\n\n", "Analysis:"])
            output = response['choices'][0]['text'].strip()
            
            # Extract JSON from output
            json_str = self._extract_json_from_output(output)
            if json_str:
                try:
                    json_data = json.loads(json_str)
                    validated_data = AnalysisAttributes(**json_data)
                    
                    # Convert to dictionary with additional metadata
                    results = validated_data.model_dump()
                    results.update({
                        'raw_output': output[:200],
                        'text': text[:100],
                        'parsing_method': 'structured_json'
                    })
                    
                    return results
                    
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"⚠️ JSON validation failed, using fallback")
                    return self._fallback_analysis(text)
            else:
                print(f"⚠️ No JSON found in output, using fallback")
                return self._fallback_analysis(text)
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return self._fallback_analysis(text)
    
    def _extract_json_from_output(self, output):
        """Extract JSON object from model output"""
        import re
        
        # Look for JSON object patterns
        json_patterns = [
            r'\{[^}]*"sentiment"[^}]*\}',
            r'\{.*?"sentiment".*?\}',
            r'\{.*?\}',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, output, re.DOTALL)
            if matches:
                # Try to parse the first match
                for match in matches:
                    try:
                        # Clean up the match
                        clean_match = match.strip()
                        json.loads(clean_match)  # Test if valid JSON
                        return clean_match
                    except json.JSONDecodeError:
                        continue
        
        return None
    
    def _get_json_grammar(self):
        """Generate JSON grammar for llama-cpp grammar enforcement"""
        # Simple JSON grammar for the expected output structure
        return '''
        root ::= object
        object ::= "{" member ("," member)* "}"
        member ::= string ":" value
        value ::= number | string | "true" | "false" | "null"
        string ::= "\\"" [^"]* "\\""
        number ::= [0-9]+ ("." [0-9]+)?
        '''
    
    def _fallback_analysis(self, text):
        """Fallback content-based analysis when structured output fails"""
        text_lower = text.lower()
        
        # Create default AnalysisAttributes instance
        try:
            # Smart fallback with keyword detection
            sentiment_score = 3.0
            respect_score = 2.0
            insult_score = 1.0
            humiliate_score = 1.0
            status_score = 2.0
            dehumanize_score = 1.0
            violence_score = 1.0
            genocide_score = 1.0
            attack_defend_score = 2.0
            confidence_score = 0.6
            
            # Keyword-based analysis
            hate_words = ['hate', 'kill', 'die', 'murder', 'eliminate', 'destroy']
            positive_words = ['love', 'respect', 'appreciate', 'admire', 'support']
            insult_words = ['stupid', 'idiot', 'moron', 'worthless', 'pathetic']
            dehumanize_words = ['animal', 'beast', 'vermin', 'pest', 'creature']
            violence_words = ['attack', 'hurt', 'harm', 'beat', 'strike', 'fight']
            
            if any(word in text_lower for word in hate_words):
                sentiment_score = 1.0
                respect_score = 1.0
                insult_score = 4.0
                attack_defend_score = 4.0
                confidence_score = 0.8
                
            if any(word in text_lower for word in positive_words):
                sentiment_score = 5.0
                respect_score = 2.0
                attack_defend_score = 1.0
                confidence_score = 0.8
                
            if any(word in text_lower for word in insult_words):
                insult_score = 3.0
                humiliate_score = 2.0
                attack_defend_score = 3.0
                
            if any(word in text_lower for word in dehumanize_words):
                dehumanize_score = 2.0
                status_score = 1.0
                
            if any(word in text_lower for word in violence_words):
                violence_score = 2.0
                attack_defend_score = 4.0
            
            # Create and validate using Pydantic
            fallback_data = AnalysisAttributes(
                sentiment=sentiment_score,
                respect=respect_score,
                insult=insult_score,
                humiliate=humiliate_score,
                status=status_score,
                dehumanize=dehumanize_score,
                violence=violence_score,
                genocide=genocide_score,
                attack_defend=attack_defend_score,
                confidence=confidence_score
            )
            
            results = fallback_data.model_dump()
            results.update({
                'raw_output': 'fallback_analysis',
                'text': text[:100],
                'parsing_method': 'keyword_fallback'
            })
            
            return results
            
        except Exception as e:
            print(f"❌ Fallback analysis error: {e}")
            # Ultimate fallback with default values
            default_data = AnalysisAttributes(
                sentiment=3.0,
                respect=2.0,
                insult=1.0,
                humiliate=1.0,
                status=2.0,
                dehumanize=1.0,
                violence=1.0,
                genocide=1.0,
                attack_defend=2.0,
                confidence=0.5
            )
            
            results = default_data.model_dump()
            results.update({
                'raw_output': 'default_fallback',
                'text': text[:100],
                'parsing_method': 'default_fallback'
            })
            
            return results

def visualize_analysis_results(results, filename_prefix="multi_attribute_analysis"):
    """Create visualization of multi-attribute analysis results"""
    print("\n🎨 Creating visualization of multi-attribute analysis...")
    
    try:
        # Extract all attributes
        sentiment_ratings = [r['sentiment'] for r in results]
        respect_ratings = [r['respect'] for r in results]
        insult_ratings = [r['insult'] for r in results]
        humiliate_ratings = [r['humiliate'] for r in results]
        status_ratings = [r['status'] for r in results]
        dehumanize_ratings = [r['dehumanize'] for r in results]
        violence_ratings = [r['violence'] for r in results]
        genocide_ratings = [r['genocide'] for r in results]
        attack_defend_ratings = [r['attack_defend'] for r in results]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Multi-Attribute Comment Analysis Results (Structured Output)', fontsize=16, fontweight='bold')
        
        # Plot 1: Sentiment distribution (1-5)
        axes[0,0].hist(sentiment_ratings, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_xlabel('Sentiment Rating (1-5)', fontsize=10)
        axes[0,0].set_ylabel('Frequency', fontsize=10)
        axes[0,0].set_title('Sentiment Distribution', fontsize=12)
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Respect distribution (1-2)
        axes[0,1].hist(respect_ratings, bins=4, alpha=0.7, color='green', edgecolor='black')
        axes[0,1].set_xlabel('Respect Rating (1-2)', fontsize=10)
        axes[0,1].set_ylabel('Frequency', fontsize=10)
        axes[0,1].set_title('Respect Distribution', fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Insult distribution (1-4)
        axes[0,2].hist(insult_ratings, bins=8, alpha=0.7, color='red', edgecolor='black')
        axes[0,2].set_xlabel('Insult Rating (1-4)', fontsize=10)
        axes[0,2].set_ylabel('Frequency', fontsize=10)
        axes[0,2].set_title('Insult Distribution', fontsize=12)
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Humiliate distribution (1-3)
        axes[1,0].hist(humiliate_ratings, bins=6, alpha=0.7, color='orange', edgecolor='black')
        axes[1,0].set_xlabel('Humiliate Rating (1-3)', fontsize=10)
        axes[1,0].set_ylabel('Frequency', fontsize=10)
        axes[1,0].set_title('Humiliate Distribution', fontsize=12)
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Attack-Defend distribution (1-4)
        axes[1,1].hist(attack_defend_ratings, bins=8, alpha=0.7, color='purple', edgecolor='black')
        axes[1,1].set_xlabel('Attack-Defend Rating (1-4)', fontsize=10)
        axes[1,1].set_ylabel('Frequency', fontsize=10)
        axes[1,1].set_title('Attack-Defend Distribution', fontsize=12)
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Binary attributes (Dehumanize, Violence, Genocide, Status)
        binary_data = {
            'Status': status_ratings,
            'Dehumanize': dehumanize_ratings,
            'Violence': violence_ratings,
            'Genocide': genocide_ratings
        }
        
        x_pos = range(len(binary_data))
        means = [sum(ratings)/len(ratings) for ratings in binary_data.values()]
        axes[1,2].bar(x_pos, means, alpha=0.7, color=['brown', 'darkred', 'crimson', 'black'])
        axes[1,2].set_xlabel('Binary Attributes', fontsize=10)
        axes[1,2].set_ylabel('Average Rating', fontsize=10)
        axes[1,2].set_title('Binary Attributes Average', fontsize=12)
        axes[1,2].set_xticks(x_pos)
        axes[1,2].set_xticklabels(binary_data.keys(), rotation=45)
        axes[1,2].grid(True, alpha=0.3)
        
        # Plot 7: Parsing method distribution
        parsing_methods = [r.get('parsing_method', 'unknown') for r in results]
        method_counts = {method: parsing_methods.count(method) for method in set(parsing_methods)}
        
        axes[2,0].pie(method_counts.values(), labels=method_counts.keys(), autopct='%1.1f%%', startangle=90)
        axes[2,0].set_title('Parsing Method Distribution', fontsize=12)
        
        # Plot 8: Summary statistics table
        axes[2,1].axis('off')
        
        # Calculate statistics for all attributes
        total = len(results)
        
        summary_data = [
            ['Attribute', 'Average', 'Std Dev'],
            ['Sentiment (1-5)', f'{sum(sentiment_ratings)/total:.2f}', f'{np.std(sentiment_ratings):.2f}'],
            ['Respect (1-2)', f'{sum(respect_ratings)/total:.2f}', f'{np.std(respect_ratings):.2f}'],
            ['Insult (1-4)', f'{sum(insult_ratings)/total:.2f}', f'{np.std(insult_ratings):.2f}'],
            ['Humiliate (1-3)', f'{sum(humiliate_ratings)/total:.2f}', f'{np.std(humiliate_ratings):.2f}'],
            ['Attack-Defend (1-4)', f'{sum(attack_defend_ratings)/total:.2f}', f'{np.std(attack_defend_ratings):.2f}'],
            ['Status (1-2)', f'{sum(status_ratings)/total:.2f}', f'{np.std(status_ratings):.2f}'],
            ['Dehumanize (1-2)', f'{sum(dehumanize_ratings)/total:.2f}', f'{np.std(dehumanize_ratings):.2f}'],
            ['Violence (1-2)', f'{sum(violence_ratings)/total:.2f}', f'{np.std(violence_ratings):.2f}'],
            ['Genocide (1-2)', f'{sum(genocide_ratings)/total:.2f}', f'{np.std(genocide_ratings):.2f}']
        ]
        
        # Create table
        table = axes[2,1].table(cellText=summary_data, cellLoc='center', loc='center',
                               colWidths=[0.4, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        
        # Style the table
        for i in range(len(summary_data)):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F2F2F2')
        
        axes[2,1].set_title('Summary Statistics', fontsize=12, pad=20)
        
        # Plot 9: Confidence distribution
        confidences = [r['confidence'] for r in results]
        axes[2,2].hist(confidences, bins=10, alpha=0.7, color='gold', edgecolor='black')
        axes[2,2].set_xlabel('Confidence (0.1-1.0)', fontsize=10)
        axes[2,2].set_ylabel('Frequency', fontsize=10)
        axes[2,2].set_title('Confidence Distribution', fontsize=12)
        axes[2,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_structured_dashboard_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved: {filename}")
        
        plt.show()
        
        return filename
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return None

def analyze_comments_multi_attribute(analyzer, comments_df, text_column='text', sample_size=None):
    """Analyze multiple attributes for a dataset of comments"""
    print("\n🎯 STARTING STRUCTURED MULTI-ATTRIBUTE ANALYSIS")
    print("=" * 50)
    
    # Prepare data
    if sample_size:
        if sample_size < len(comments_df):
            comments_sample = comments_df.sample(n=sample_size, random_state=42)
            print(f"📊 Analyzing random sample: {sample_size} comments")
        else:
            comments_sample = comments_df
            print(f"📊 Analyzing all comments: {len(comments_sample)}")
    else:
        comments_sample = comments_df
        print(f"📊 Analyzing all comments: {len(comments_sample)}")
    
    results = []
    total = len(comments_sample)
    
    print(f"🚀 Processing {total} comments with structured JSON output...")
    print("Analyzing: Sentiment, Respect, Insult, Humiliate, Status, Dehumanize, Violence, Genocide, Attack-Defend")
    
    for i, (_, row) in enumerate(comments_sample.iterrows(), 1):
        try:
            text = str(row[text_column])
            if len(text.strip()) == 0:
                continue
                
            result = analyzer.analyze_with_gguf(text)
            
            # Add metadata
            result.update({
                'comment_id': row.get('comment_id', i),
                'index': i
            })
            
            results.append(result)
            
            # Progress update
            if i % 5 == 0 or i == total:
                avg_sentiment = sum(r['sentiment'] for r in results) / len(results)
                avg_confidence = sum(r['confidence'] for r in results) / len(results)
                structured_count = sum(1 for r in results if r.get('parsing_method') == 'structured_json')
                print(f"  Progress: {i}/{total} ({i/total*100:.1f}%) | Avg Sentiment: {avg_sentiment:.2f} | Avg Confidence: {avg_confidence:.2f} | Structured: {structured_count}/{len(results)}")
                
        except Exception as e:
            print(f"  ⚠️ Error processing comment {i}: {e}")
            continue
            
        # Small delay to prevent overheating
        time.sleep(0.1)
    
    print(f"✅ Structured multi-attribute analysis complete! Processed {len(results)} comments")
    return results

def save_results(results, prefix="multi_attribute_structured"):
    """Save analysis results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    df = pd.DataFrame(results)
    csv_file = f"{prefix}_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"💾 Results saved: {csv_file}")
    
    # Save as JSON for detailed analysis
    json_file = f"{prefix}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Detailed results saved: {json_file}")
    
    return csv_file, json_file

def main():
    """Main function to run structured multi-attribute comment analysis"""
    print("🎭 LLAMA-CPP STRUCTURED MULTI-ATTRIBUTE ANALYZER")
    print("=" * 60)
    print("Features: Pydantic validation, JSON schema enforcement, robust parsing")
    print("Analyzing: Sentiment, Respect, Insult, Humiliate, Status,")
    print("          Dehumanize, Violence, Genocide, Attack-Defend")
    
    # Initialize model
    analyzer = initialize_llamacpp()
    if not analyzer:
        print("❌ Failed to initialize model")
        return
    
    # Load data
    print("\n📂 LOADING DATA")
    print("=" * 20)
    
    # Try to load the dataset
    dataset_files = [
        'selected_comments.csv',
        'processed_dataset.csv', 
        'comments.csv'
    ]
    
    df = None
    for file in dataset_files:
        if os.path.exists(file):
            try:
                print(f"📄 Loading: {file}")
                df = pd.read_csv(file)
                print(f"✅ Loaded {len(df)} rows")
                break
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
                continue
    
    if df is None:
        print("❌ No dataset found. Expected files:")
        for file in dataset_files:
            print(f"  • {file}")
        return
    
    # Determine text column
    text_columns = ['text', 'comment', 'content', 'message']
    text_column = None
    for col in text_columns:
        if col in df.columns:
            text_column = col
            break
    
    if not text_column:
        print(f"❌ No text column found. Available columns: {list(df.columns)}")
        return
    
    print(f"📝 Using text column: '{text_column}'")
    
    # Ask for sample size
    total_comments = len(df)
    print(f"\n📊 Dataset contains {total_comments} comments")
    
    while True:
        try:
            sample_input = input(f"Analyze how many comments? (max {total_comments}, or 'all'): ").strip()
            if sample_input.lower() in ['all', 'a']:
                sample_size = None
                break
            else:
                sample_size = int(sample_input)
                if 1 <= sample_size <= total_comments:
                    break
                else:
                    print(f"Please enter a number between 1 and {total_comments}")
        except ValueError:
            print("Please enter a valid number or 'all'")
    
    # Run analysis
    results = analyze_comments_multi_attribute(analyzer, df, text_column, sample_size)
    
    if not results:
        print("❌ No results generated")
        return
    
    # Save results
    print(f"\n💾 SAVING RESULTS")
    print("=" * 20)
    csv_file, json_file = save_results(results)
    
    # Create visualization
    print(f"\n🎨 CREATING VISUALIZATION")
    print("=" * 25)
    plot_file = visualize_analysis_results(results)
    
    # Display summary
    print(f"\n📊 STRUCTURED MULTI-ATTRIBUTE ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Extract attributes for summary
    sentiment_ratings = [r['sentiment'] for r in results]
    respect_ratings = [r['respect'] for r in results]
    insult_ratings = [r['insult'] for r in results]
    humiliate_ratings = [r['humiliate'] for r in results]
    attack_defend_ratings = [r['attack_defend'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # Count parsing methods
    parsing_methods = [r.get('parsing_method', 'unknown') for r in results]
    structured_count = parsing_methods.count('structured_json')
    fallback_count = parsing_methods.count('keyword_fallback')
    
    print(f"📈 Total comments analyzed: {len(results)}")
    print(f"📈 Structured JSON parsing: {structured_count} ({structured_count/len(results)*100:.1f}%)")
    print(f"📈 Fallback parsing: {fallback_count} ({fallback_count/len(results)*100:.1f}%)")
    print(f"📈 Average confidence: {sum(confidences)/len(confidences):.3f}")
    
    print(f"\n📊 Attribute Averages:")
    print(f"  • Sentiment (1-5): {sum(sentiment_ratings)/len(sentiment_ratings):.2f}")
    print(f"  • Respect (1-2): {sum(respect_ratings)/len(respect_ratings):.2f}")
    print(f"  • Insult (1-4): {sum(insult_ratings)/len(insult_ratings):.2f}")
    print(f"  • Humiliate (1-3): {sum(humiliate_ratings)/len(humiliate_ratings):.2f}")
    print(f"  • Attack-Defend (1-4): {sum(attack_defend_ratings)/len(attack_defend_ratings):.2f}")
    
    print(f"\n✅ Structured multi-attribute analysis complete!")
    print(f"📁 CSV: {csv_file}")
    print(f"📁 JSON: {json_file}")
    if plot_file:
        print(f"📁 Dashboard: {plot_file}")

if __name__ == "__main__":
    main()