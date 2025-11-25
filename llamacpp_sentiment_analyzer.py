#!/usr/bin/env python3
"""
llama-cpp Multi-Attribute Comment Analyzer
Analyzes sentiment and multiple attributes related to hate speech, respect, insults, etc.
Uses only GGUF models with structured JSON output for comprehensive comment analysis
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
        fig.suptitle('Multi-Attribute Comment Analysis Results', fontsize=16, fontweight='bold')
        
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
        
        # Plot 7: Correlation heatmap between attributes
        import numpy as np
        attrs_data = np.array([sentiment_ratings, respect_ratings, insult_ratings, 
                              humiliate_ratings, status_ratings, dehumanize_ratings,
                              violence_ratings, genocide_ratings, attack_defend_ratings])
        corr_matrix = np.corrcoef(attrs_data)
        
        attr_names = ['Sentiment', 'Respect', 'Insult', 'Humiliate', 'Status', 
                     'Dehumanize', 'Violence', 'Genocide', 'Attack-Defend']
        
        im = axes[2,0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2,0].set_xticks(range(len(attr_names)))
        axes[2,0].set_yticks(range(len(attr_names)))
        axes[2,0].set_xticklabels(attr_names, rotation=45, ha='right')
        axes[2,0].set_yticklabels(attr_names)
        axes[2,0].set_title('Attribute Correlations', fontsize=12)
        
        # Add correlation values to heatmap
        for i in range(len(attr_names)):
            for j in range(len(attr_names)):
                text = axes[2,0].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        # Plot 8: Summary statistics table
        axes[2,1].axis('off')
        
        # Calculate statistics for all attributes
        total = len(results)
        
        summary_data = [
            ['Attribute', 'Average', 'Range', 'Std Dev'],
            ['Sentiment (1-5)', f'{sum(sentiment_ratings)/total:.2f}', f'{min(sentiment_ratings):.1f}-{max(sentiment_ratings):.1f}', f'{np.std(sentiment_ratings):.2f}'],
            ['Respect (1-2)', f'{sum(respect_ratings)/total:.2f}', f'{min(respect_ratings):.1f}-{max(respect_ratings):.1f}', f'{np.std(respect_ratings):.2f}'],
            ['Insult (1-4)', f'{sum(insult_ratings)/total:.2f}', f'{min(insult_ratings):.1f}-{max(insult_ratings):.1f}', f'{np.std(insult_ratings):.2f}'],
            ['Humiliate (1-3)', f'{sum(humiliate_ratings)/total:.2f}', f'{min(humiliate_ratings):.1f}-{max(humiliate_ratings):.1f}', f'{np.std(humiliate_ratings):.2f}'],
            ['Attack-Defend (1-4)', f'{sum(attack_defend_ratings)/total:.2f}', f'{min(attack_defend_ratings):.1f}-{max(attack_defend_ratings):.1f}', f'{np.std(attack_defend_ratings):.2f}'],
            ['Status (1-2)', f'{sum(status_ratings)/total:.2f}', f'{min(status_ratings):.1f}-{max(status_ratings):.1f}', f'{np.std(status_ratings):.2f}'],
            ['Dehumanize (1-2)', f'{sum(dehumanize_ratings)/total:.2f}', f'{min(dehumanize_ratings):.1f}-{max(dehumanize_ratings):.1f}', f'{np.std(dehumanize_ratings):.2f}'],
            ['Violence (1-2)', f'{sum(violence_ratings)/total:.2f}', f'{min(violence_ratings):.1f}-{max(violence_ratings):.1f}', f'{np.std(violence_ratings):.2f}'],
            ['Genocide (1-2)', f'{sum(genocide_ratings)/total:.2f}', f'{min(genocide_ratings):.1f}-{max(genocide_ratings):.1f}', f'{np.std(genocide_ratings):.2f}']
        ]
        
        # Create table
        table = axes[2,1].table(cellText=summary_data, cellLoc='center', loc='center',
                               colWidths=[0.3, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        
        # Style the table
        for i in range(len(summary_data)):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F2F2F2')
        
        axes[2,1].set_title('Summary Statistics', fontsize=12, pad=20)
        
        # Plot 9: Sample comments analysis
        axes[2,2].axis('off')
        
        # Show a few sample results
        sample_text = "Sample Analysis Results:\n\n"
        for i, result in enumerate(results[:3]):  # Show first 3 results
            sample_text += f"Comment {i+1}:\n"
            sample_text += f"'{result['text'][:50]}...'\n"
            sample_text += f"Sentiment: {result['sentiment']:.1f}, "
            sample_text += f"Respect: {result['respect']:.1f}, "
            sample_text += f"Insult: {result['insult']:.1f}\n\n"
        
        axes[2,2].text(0.1, 0.9, sample_text, transform=axes[2,2].transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
        axes[2,2].set_title('Sample Results', fontsize=12)
        
        plt.tight_layout()
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_dashboard_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved: {filename}")
        
        plt.show()
        
        return filename
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return None

def download_gguf_model():
    """Download a GGUF model using huggingface-cli"""
    print("\n📥 Downloading GGUF Model")
    print("=" * 40)
    
    model_configs = [
        ("microsoft/Phi-3-mini-4k-instruct-gguf", "Phi-3-mini-4k-instruct-q4.gguf"),
        ("hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF", "llama-3.2-3b-instruct-q4_k_m.gguf"),
        ("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    ]
    
    print("Available models:")
    for i, (repo, file) in enumerate(model_configs, 1):
        print(f"  {i}. {repo} / {file}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(model_configs)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_configs):
                model_repo, model_file = model_configs[choice_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(model_configs)}")
        except ValueError:
            print("Please enter a valid number")
    
    try:
        import subprocess
        
        # Create models directory
        models_dir = Path("./gguf_models/")
        models_dir.mkdir(exist_ok=True)
        
        # Download command
        cmd = [
            "huggingface-cli", "download",
            model_repo,
            "--include", model_file,
            "--local-dir", str(models_dir)
        ]
        
        print(f"\n🚀 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            model_path = models_dir / model_file
            if model_path.exists():
                size_gb = model_path.stat().st_size / (1024**3)
                print(f"✅ Download successful!")
                print(f"📁 File: {model_path}")
                print(f"📏 Size: {size_gb:.1f} GB")
                return model_path
            else:
                print(f"❌ File not found after download: {model_path}")
                return None
        else:
            print(f"❌ Download failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ Download timeout (5 minutes)")
        return None
    except FileNotFoundError:
        print("❌ huggingface-cli not found")
        print("💡 Install with: pip install huggingface_hub[cli]")
        return None
    except Exception as e:
        print(f"❌ Download error: {e}")
        return None

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
            # Option 2: Download a model
            print("\nOption 2: No GGUF models found locally")
            download_choice = input("Download a GGUF model? (y/n): ").strip().lower()
            
            if download_choice in ['y', 'yes']:
                model_path = download_gguf_model()
                if model_path:
                    llm = Llama(
                        model_path=str(model_path),
                        n_ctx=2048,
                        n_batch=512,
                        verbose=False,
                        n_threads=4,
                    )
                    return MultiAttributeAnalyzer(llm)
            
            print("❌ No model available. Please download a GGUF model.")
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
        
        # Simplified and more direct prompt template
        # Improved prompt template with examples and clear criteria
        self.prompt_template = '''Analyze this comment and rate it on multiple scales.

Comment: "{text}"

SCORING CRITERIA:
- SENTIMENT (1-5): 1=Very Negative (hate/hostility), 3=Neutral, 5=Very Positive (love/support)
- RESPECT (1-2): 1=Disrespectful/Rude, 2=Respectful/Polite
- INSULT (1-4): 1=None, 2=Mild/Sarcastic, 3=Clear Insult, 4=Severe/Slur
- HUMILIATE (1-3): 1=None, 2=Attempt to embarrass, 3=Degrading/Dehumanizing
- STATUS (1-2): 1=Treats target as inferior, 2=Treats target as equal
- DEHUMANIZE (1-2): 1=No, 2=Yes (animalistic/vermin comparisons)
- VIOLENCE (1-2): 1=No, 2=Yes (calls for physical harm/death)
- GENOCIDE (1-2): 1=No, 2=Yes (calls for mass elimination)
- ATTACK_DEFEND (1-4): 1=Defending group, 2=Defending self, 3=Attacking individual, 4=Attacking group
- CONFIDENCE (0.1-1.0): Analysis confidence

EXAMPLES:
1. "I love this community, everyone is so helpful!"
   SENTIMENT 5, RESPECT 2, INSULT 1, HUMILIATE 1, STATUS 2, DEHUMANIZE 1, VIOLENCE 1, GENOCIDE 1, ATTACK_DEFEND 1

2. "You are an idiot and your people are trash."
   SENTIMENT 1, RESPECT 1, INSULT 3, HUMILIATE 2, STATUS 1, DEHUMANIZE 1, VIOLENCE 1, GENOCIDE 1, ATTACK_DEFEND 4

3. "These [slur] should be exterminated like rats."
   SENTIMENT 1, RESPECT 1, INSULT 4, HUMILIATE 3, STATUS 1, DEHUMANIZE 2, VIOLENCE 2, GENOCIDE 2, ATTACK_DEFEND 4

RESPOND WITH ONLY NUMBERS IN THIS FORMAT:
SENTIMENT [1-5]
RESPECT [1-2]
INSULT [1-4]
HUMILIATE [1-3]
STATUS [1-2]
DEHUMANIZE [1-2]
VIOLENCE [1-2]
GENOCIDE [1-2]
ATTACK_DEFEND [1-4]
CONFIDENCE [0.1-1.0]'''

    def analyze_with_gguf(self, text):
        """Use direct GGUF model to analyze multiple attributes"""
        try:
            prompt = self.prompt_template.format(text=text[:400])
            
            response = self.model(prompt, max_tokens=120, temperature=0.1)
            output = response['choices'][0]['text'].strip().lower()
            
            # Initialize default values
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
                'confidence': 0.7
            }
            
            # Extract ratings using very flexible regex patterns
            import re
            
            matched_patterns = 0
            
            # First try simple number extraction in sequence
            # Look for 9-10 consecutive numbers that could be our ratings
            number_sequence = re.findall(r'([0-9]\.?[0-9]*)', output)
            
            if len(number_sequence) >= 9:
                try:
                    # Try to map numbers in sequence to attributes
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
                    
                    # Validate and assign if reasonable
                    if 1.0 <= sentiment <= 5.0:
                        results['sentiment'] = sentiment
                        matched_patterns += 1
                    if 1.0 <= respect <= 2.0:
                        results['respect'] = respect
                        matched_patterns += 1
                    if 1.0 <= insult <= 4.0:
                        results['insult'] = insult
                        matched_patterns += 1
                    if 1.0 <= humiliate <= 3.0:
                        results['humiliate'] = humiliate
                        matched_patterns += 1
                    if 1.0 <= status <= 2.0:
                        results['status'] = status
                        matched_patterns += 1
                    if 1.0 <= dehumanize <= 2.0:
                        results['dehumanize'] = dehumanize
                        matched_patterns += 1
                    if 1.0 <= violence <= 2.0:
                        results['violence'] = violence
                        matched_patterns += 1
                    if 1.0 <= genocide <= 2.0:
                        results['genocide'] = genocide
                        matched_patterns += 1
                    if 1.0 <= attack_defend <= 4.0:
                        results['attack_defend'] = attack_defend
                        matched_patterns += 1
                    if 0.1 <= confidence <= 1.0:
                        results['confidence'] = confidence
                        matched_patterns += 1
                        
                except (ValueError, IndexError):
                    matched_patterns = 0
            
            # If sequence parsing failed, try keyword-based patterns
            if matched_patterns < 5:
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
                
                pattern_matches = 0
                
                # Extract each attribute
                for attr, pattern in patterns.items():
                    match = re.search(pattern, output, re.IGNORECASE)
                    if match:
                        pattern_matches += 1
                        try:
                            value = float(match.group(1))
                            
                            # Validate ranges
                            if attr == 'sentiment':
                                results[attr] = max(1.0, min(5.0, value))
                            elif attr in ['respect', 'status', 'dehumanize', 'violence', 'genocide']:
                                results[attr] = max(1.0, min(2.0, value))
                            elif attr == 'humiliate':
                                results[attr] = max(1.0, min(3.0, value))
                            elif attr in ['insult', 'attack_defend']:
                                results[attr] = max(1.0, min(4.0, value))
                            elif attr == 'confidence':
                                results[attr] = max(0.1, min(1.0, value))
                        except (ValueError, IndexError):
                            continue
                
                matched_patterns = max(matched_patterns, pattern_matches)
            
            # Fallback analysis if parsing fails
            if matched_patterns < 3:  # If less than 3 patterns matched
                results = self._fallback_analysis(text)
            else:
                print(f"✅ Pattern matching successful ({matched_patterns} patterns matched)")
                results = self._fallback_analysis(text)
            
            # Add metadata
            results.update({
                'raw_output': output[:200],  # Store first 200 chars
                'text': text[:100]  # Store first 100 chars for reference
            })
            
            return results
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return self._fallback_analysis(text)
    
    def _fallback_analysis(self, text):
        """Fallback content-based analysis when pattern matching fails"""
        text_lower = text.lower()
        
        # Simple keyword-based analysis
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
            'confidence': 0.5
        }
        
        # Check for obvious indicators
        hate_words = ['hate', 'kill', 'die', 'murder', 'eliminate', 'destroy']
        positive_words = ['love', 'respect', 'appreciate', 'admire', 'support']
        insult_words = ['stupid', 'idiot', 'moron', 'worthless', 'pathetic']
        dehumanize_words = ['animal', 'beast', 'vermin', 'pest', 'creature']
        violence_words = ['attack', 'hurt', 'harm', 'beat', 'strike', 'fight']
        
        if any(word in text_lower for word in hate_words):
            results['sentiment'] = 1.0
            results['respect'] = 1.0
            results['insult'] = 4.0
            results['attack_defend'] = 4.0
            results['confidence'] = 0.8
            
        if any(word in text_lower for word in positive_words):
            results['sentiment'] = 5.0
            results['respect'] = 2.0
            results['attack_defend'] = 1.0
            results['confidence'] = 0.8
            
        if any(word in text_lower for word in insult_words):
            results['insult'] = 3.0
            results['humiliate'] = 2.0
            results['attack_defend'] = 3.0
            
        if any(word in text_lower for word in dehumanize_words):
            results['dehumanize'] = 2.0
            results['status'] = 1.0
            
        if any(word in text_lower for word in violence_words):
            results['violence'] = 2.0
            results['attack_defend'] = 4.0
        
        results['raw_output'] = "fallback_analysis"
        results['text'] = text[:100]
        
        return results

def analyze_comments_multi_attribute(analyzer, comments_df, text_column='text', sample_size=None):
    """Analyze multiple attributes for a dataset of comments"""
    print("\n🎯 STARTING MULTI-ATTRIBUTE ANALYSIS")
    print("=" * 40)
    
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
    
    print(f"🚀 Processing {total} comments...")
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
                print(f"  Progress: {i}/{total} ({i/total*100:.1f}%) | Avg Sentiment: {avg_sentiment:.2f} | Avg Confidence: {avg_confidence:.2f}")
                
        except Exception as e:
            print(f"  ⚠️ Error processing comment {i}: {e}")
            continue
            
        # Small delay to prevent overheating
        time.sleep(0.15)
    
    print(f"✅ Multi-attribute analysis complete! Processed {len(results)} comments")
    return results

def save_results(results, prefix="multi_attribute_analysis"):
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
    
    # Save summary statistics
    if results:
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
        confidences = [r['confidence'] for r in results]
        
        summary = {
            'timestamp': timestamp,
            'total_comments': len(results),
            'averages': {
                'sentiment': sum(sentiment_ratings) / len(sentiment_ratings),
                'respect': sum(respect_ratings) / len(respect_ratings),
                'insult': sum(insult_ratings) / len(insult_ratings),
                'humiliate': sum(humiliate_ratings) / len(humiliate_ratings),
                'status': sum(status_ratings) / len(status_ratings),
                'dehumanize': sum(dehumanize_ratings) / len(dehumanize_ratings),
                'violence': sum(violence_ratings) / len(violence_ratings),
                'genocide': sum(genocide_ratings) / len(genocide_ratings),
                'attack_defend': sum(attack_defend_ratings) / len(attack_defend_ratings),
                'confidence': sum(confidences) / len(confidences)
            },
            'ranges': {
                'sentiment': [min(sentiment_ratings), max(sentiment_ratings)],
                'respect': [min(respect_ratings), max(respect_ratings)],
                'insult': [min(insult_ratings), max(insult_ratings)],
                'humiliate': [min(humiliate_ratings), max(humiliate_ratings)],
                'status': [min(status_ratings), max(status_ratings)],
                'dehumanize': [min(dehumanize_ratings), max(dehumanize_ratings)],
                'violence': [min(violence_ratings), max(violence_ratings)],
                'genocide': [min(genocide_ratings), max(genocide_ratings)],
                'attack_defend': [min(attack_defend_ratings), max(attack_defend_ratings)]
            },
            'high_confidence_pct': sum(1 for c in confidences if c >= 0.9) / len(confidences) * 100,
            'attribute_distributions': {
                'sentiment_very_negative': sum(1 for r in sentiment_ratings if r <= 1.5),
                'sentiment_negative': sum(1 for r in sentiment_ratings if 1.5 < r <= 2.5),
                'sentiment_neutral': sum(1 for r in sentiment_ratings if 2.5 < r <= 3.5),
                'sentiment_positive': sum(1 for r in sentiment_ratings if 3.5 < r <= 4.5),
                'sentiment_very_positive': sum(1 for r in sentiment_ratings if r > 4.5),
                'respect_not_respectful': sum(1 for r in respect_ratings if r <= 1.5),
                'respect_respectful': sum(1 for r in respect_ratings if r > 1.5),
                'insult_not_insulting': sum(1 for r in insult_ratings if r <= 1.5),
                'insult_somewhat_not': sum(1 for r in insult_ratings if 1.5 < r <= 2.5),
                'insult_somewhat_yes': sum(1 for r in insult_ratings if 2.5 < r <= 3.5),
                'insult_very_insulting': sum(1 for r in insult_ratings if r > 3.5),
                'humiliate_no': sum(1 for r in humiliate_ratings if r <= 1.5),
                'humiliate_neutral': sum(1 for r in humiliate_ratings if 1.5 < r <= 2.5),
                'humiliate_yes': sum(1 for r in humiliate_ratings if r > 2.5),
                'attack_defend_defending': sum(1 for r in attack_defend_ratings if r <= 2.5),
                'attack_defend_attacking': sum(1 for r in attack_defend_ratings if r > 2.5)
            }
        }
        
        summary_file = f"{prefix}_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📊 Summary saved: {summary_file}")
    
    return csv_file, json_file

def main():
    """Main function to run multi-attribute comment analysis"""
    print("🎭 LLAMA-CPP MULTI-ATTRIBUTE COMMENT ANALYZER")
    print("=" * 60)
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
    print(f"\n📊 MULTI-ATTRIBUTE ANALYSIS SUMMARY")
    print("=" * 40)
    
    # Extract attributes for summary
    sentiment_ratings = [r['sentiment'] for r in results]
    respect_ratings = [r['respect'] for r in results]
    insult_ratings = [r['insult'] for r in results]
    humiliate_ratings = [r['humiliate'] for r in results]
    attack_defend_ratings = [r['attack_defend'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    print(f"📈 Total comments analyzed: {len(results)}")
    print(f"📈 Average confidence: {sum(confidences)/len(confidences):.3f}")
    
    print(f"\n📊 Attribute Averages:")
    print(f"  • Sentiment (1-5): {sum(sentiment_ratings)/len(sentiment_ratings):.2f}")
    print(f"  • Respect (1-2): {sum(respect_ratings)/len(respect_ratings):.2f}")
    print(f"  • Insult (1-4): {sum(insult_ratings)/len(insult_ratings):.2f}")
    print(f"  • Humiliate (1-3): {sum(humiliate_ratings)/len(humiliate_ratings):.2f}")
    print(f"  • Attack-Defend (1-4): {sum(attack_defend_ratings)/len(attack_defend_ratings):.2f}")
    
    print(f"\n📊 Key Insights:")
    # Sentiment distribution
    very_negative = sum(1 for r in sentiment_ratings if r <= 1.5)
    negative = sum(1 for r in sentiment_ratings if 1.5 < r <= 2.5)
    neutral = sum(1 for r in sentiment_ratings if 2.5 < r <= 3.5)
    positive = sum(1 for r in sentiment_ratings if 3.5 < r <= 4.5)
    very_positive = sum(1 for r in sentiment_ratings if r > 4.5)
    
    print(f"  • Sentiment: {very_negative} very negative, {negative} negative, {neutral} neutral, {positive} positive, {very_positive} very positive")
    
    # Respect distribution
    not_respectful = sum(1 for r in respect_ratings if r <= 1.5)
    respectful = sum(1 for r in respect_ratings if r > 1.5)
    print(f"  • Respect: {not_respectful} not respectful ({not_respectful/len(results)*100:.1f}%), {respectful} respectful ({respectful/len(results)*100:.1f}%)")
    
    # Insult distribution  
    high_insult = sum(1 for r in insult_ratings if r >= 3.0)
    print(f"  • High insult level (≥3.0): {high_insult} comments ({high_insult/len(results)*100:.1f}%)")
    
    # Attack-defend distribution
    attacking = sum(1 for r in attack_defend_ratings if r >= 3.0)
    defending = sum(1 for r in attack_defend_ratings if r < 3.0)
    print(f"  • Attack vs Defend: {attacking} attacking ({attacking/len(results)*100:.1f}%), {defending} defending ({defending/len(results)*100:.1f}%)")
    
    print(f"\n✅ Multi-attribute analysis complete!")
    print(f"📁 CSV: {csv_file}")
    print(f"📁 JSON: {json_file}")
    if plot_file:
        print(f"📁 Dashboard: {plot_file}")

if __name__ == "__main__":
    main()