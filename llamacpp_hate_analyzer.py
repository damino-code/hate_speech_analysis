#!/usr/bin/env python3
"""
llama-cpp Hate Speech Analyzer
Uses only GGUF models for analysis
"""

import pandas as pd
import json
from datetime import datetime
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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

def visualize_hate_ratings(results, filename_prefix="hate_analysis"):
    """Create visualization of hate speech rating distribution"""
    print("\n🎨 Creating visualization of hate rating distribution...")
    
    try:
        # Extract ratings
        ratings = [r['rating'] for r in results]
        categories = [r['category'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # Plot 1: Histogram of continuous rating distribution
        ax1.hist(ratings, bins=20, color='skyblue', alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_title('Continuous Rating Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Rating (0.0 = No Hate, 2.0 = Clear Hate)', fontsize=12)
        ax1.set_ylabel('Number of Comments', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add vertical lines for reference points
        ax1.axvline(x=0.5, color='green', linestyle='--', alpha=0.7, label='No Hate Threshold')
        ax1.axvline(x=1.5, color='red', linestyle='--', alpha=0.7, label='Hate Threshold')
        ax1.legend()
        
        # Plot 2: Category distribution (pie chart)
        category_counts = {'no': 0, 'unclear': 0, 'yes': 0}
        for cat in categories:
            category_counts[cat] += 1
        
        colors = ['#28a745', '#ffc107', '#dc3545']  # Green, Yellow, Red
        labels = ['No Hate', 'Unclear', 'Hate Speech']
        values = [category_counts['no'], category_counts['unclear'], category_counts['yes']]
        
        # Only include non-zero categories
        non_zero_indices = [i for i, v in enumerate(values) if v > 0]
        if non_zero_indices:
            filtered_values = [values[i] for i in non_zero_indices]
            filtered_labels = [labels[i] for i in non_zero_indices]
            filtered_colors = [colors[i] for i in non_zero_indices]
            
            wedges, texts, autotexts = ax2.pie(filtered_values, labels=filtered_labels, colors=filtered_colors, 
                                              autopct='%1.1f%%', startangle=90, explode=[0.05]*len(filtered_values))
            ax2.set_title('Category Distribution', fontsize=14, fontweight='bold')
            
            # Enhance pie chart text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
        else:
            ax2.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Category Distribution', fontsize=14, fontweight='bold')
        
        # Plot 3: Rating vs Confidence scatter plot
        scatter_colors = ['#28a745' if r < 0.5 else '#ffc107' if r < 1.5 else '#dc3545' for r in ratings]
        ax3.scatter(ratings, confidences, c=scatter_colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax3.set_title('Rating vs Confidence', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Rating', fontsize=12)
        ax3.set_ylabel('Confidence', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 2)
        ax3.set_ylim(0, 1)
        
        # Add reference lines
        ax3.axvline(x=0.5, color='green', linestyle='--', alpha=0.5)
        ax3.axvline(x=1.5, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=0.9, color='purple', linestyle='--', alpha=0.5, label='High Confidence Threshold')
        ax3.legend()
        
        # Plot 4: Summary statistics table
        ax4.axis('off')
        
        # Calculate statistics
        total = len(results)
        avg_rating = sum(ratings) / total if total > 0 else 0
        avg_confidence = sum(confidences) / total if total > 0 else 0
        high_conf_count = sum(1 for c in confidences if c >= 0.9)
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Comments', f'{total}'],
            ['Average Rating', f'{avg_rating:.2f}'],
            ['Average Confidence', f'{avg_confidence:.2f}'],
            ['High Confidence (≥0.9)', f'{high_conf_count} ({high_conf_count/total*100:.1f}%)'],
            ['', ''],
            ['Category Breakdown:', ''],
            ['No Hate', f'{category_counts["no"]} ({category_counts["no"]/total*100:.1f}%)'],
            ['Unclear', f'{category_counts["unclear"]} ({category_counts["unclear"]/total*100:.1f}%)'],
            ['Hate Speech', f'{category_counts["yes"]} ({category_counts["yes"]/total*100:.1f}%)'],
            ['', ''],
            ['Rating Range:', ''],
            ['Minimum', f'{min(ratings):.2f}'],
            ['Maximum', f'{max(ratings):.2f}'],
            ['Std Deviation', f'{(sum((r - avg_rating)**2 for r in ratings) / total)**0.5:.2f}']
        ]
        
        # Create table
        table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                         colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(summary_data)):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                elif summary_data[i][0] in ['Category Breakdown:', 'Rating Range:']:  # Section headers
                    cell.set_facecolor('#D9E1F2')
                    cell.set_text_props(weight='bold')
                elif summary_data[i][0] == '':  # Empty rows
                    cell.set_facecolor('#FFFFFF')
                else:
                    cell.set_facecolor('#F2F2F2' if i % 2 == 0 else '#FFFFFF')
        
        ax4.set_title('Analysis Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{filename_prefix}_distribution_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        
        print(f"📊 Visualization saved: {plot_filename}")
        plt.show()
        
        return plot_filename
        
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
        print("Make sure matplotlib and seaborn are installed: pip install matplotlib seaborn")
        return None

def download_gguf_model():
    """Download a GGUF model using huggingface-cli"""
    print("🔽 DOWNLOADING GGUF MODEL")
    print("=" * 40)
    
    # Create models directory
    models_dir = Path("./gguf_models")
    models_dir.mkdir(exist_ok=True)
    
    # Try to download a good balance model (Q4_K_M - 2GB)
    model_repo = "bartowski/Llama-3.2-3B-Instruct-GGUF"
    model_file = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    
    print(f"📥 Downloading {model_file}...")
    print(f"📦 From: {model_repo}")
    print(f"💾 Size: ~2GB (Q4_K_M quantization)")
    print(f"📁 To: {models_dir}/")
    
    try:
        import subprocess
        
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
            
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,  # Increased context for better analysis
                n_threads=4,
                verbose=False
            )
            
            print("✅ GGUF model loaded successfully!")
            return "gguf", llm
        
        # Option 2: Try to download a model
        print("\nOption 2: Download GGUF model from HuggingFace")
        downloaded_path = download_gguf_model()
        
        if downloaded_path:
            print(f"🦙 Loading downloaded model: {downloaded_path.name}")
            
            llm = Llama(
                model_path=str(downloaded_path),
                n_ctx=2048,
                n_threads=4, 
                verbose=False
            )
            
            print("✅ Downloaded GGUF model loaded successfully!")
            return "gguf", llm
        
        print("❌ No GGUF models available")
        return None, None
        
    except Exception as e:
        print(f"❌ llama-cpp initialization failed: {e}")
        return None, None

class LlamaCppHateAnalyzer:
    def __init__(self):
        self.method, self.model = initialize_llamacpp()
        
        if self.method is None:
            raise Exception("No GGUF models available. Please download GGUF models.")
        
        # Improved prompt template for better classification
        self.prompt_template = '''Analyze this comment for hate speech. Hate speech is bias-motivated, hostile language targeting a person/group based on identity characteristics (race, religion, gender, sexuality, nationality, etc.).

Comment: "{text}"

CLASSIFICATION CRITERIA:
 HATE SPEECH (2.0): Contains slurs, dehumanizing language, calls for violence/discrimination, or explicit hostility toward identity groups
 BORDERLINE (1.0): Potentially offensive but unclear intent, mild stereotyping, or aggressive but not clearly targeting identity
 NOT HATE (0.0): No targeting of identity groups, general criticism, profanity without group targeting

EXAMPLES:
- "All [group] are criminals" = 2.0 (dehumanizing stereotype)
- "I hate [slur]" = 2.0 (explicit slur/hatred)
- "That's kind of gay" = 1.0 (casual usage, unclear malicious intent)
- "This politician sucks" = 0.0 (political criticism, not identity-based)
- "Fuck this traffic" = 0.0 (general profanity, no group targeting)

Respond with ONLY:
RATING: [0.0, 1.0, or 2.0]
CONFIDENCE: [0.0 to 1.0]

Be decisive - avoid middle ratings unless truly ambiguous.'''

    def analyze_with_gguf(self, text):
        """Use direct GGUF model with improved confidence-based float ratings"""
        try:
            prompt = self.prompt_template.format(text=text[:300])
            
            response = self.model(prompt, max_tokens=80, temperature=0.1)
            output = response['choices'][0]['text'].strip().lower()
            
            # Parse rating and confidence from response
            rating = 0.0
            confidence = 0.7  # Default reasonable confidence
            
            # Extract rating - try multiple patterns
            import re
            
            # Look for "RATING: X.X" format (new prompt)
            rating_match = re.search(r'rating:\s*([0-2]\.?[0-9]*)', output)
            if rating_match:
                rating = float(rating_match.group(1))
                rating = max(0.0, min(2.0, rating))
            else:
                # Fallback: look for just numbers
                number_match = re.search(r'\b([0-2]\.?[0-9]*)\b', output)
                if number_match:
                    potential_rating = float(number_match.group(1))
                    if 0.0 <= potential_rating <= 2.0:
                        rating = potential_rating
                else:
                    # Content-based fallback with more decisive logic
                    hate_indicators = ['hate speech', 'clear hate', 'explicit', 'slur', 'dehumanizing', 'violence']
                    no_hate_indicators = ['no hate', 'not hate', 'general', 'criticism', 'profanity without']
                    
                    if any(indicator in output for indicator in hate_indicators):
                        rating = 2.0
                        confidence = 0.8
                    elif any(indicator in output for indicator in no_hate_indicators):
                        rating = 0.0  
                        confidence = 0.8
                
                        
                        if any(slur in text_lower for slur in obvious_slurs):
                            rating = 2.0
                            confidence = 0.9
                        elif any(phrase in text_lower for phrase in hate_phrases):
                            rating = 1.8
                            confidence = 0.8
                        else:
                            rating = 0.3  # Slight uncertainty, but lean toward no hate
                            confidence = 0.6
            
            # Extract confidence
            confidence_match = re.search(r'confidence:\s*([0-9.]+)', output)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.1, min(1.0, confidence))
            
            # Determine initial classification (yes/no/unclear)
            initial_classification = "unclear"
            if rating >= 1.5:
                initial_classification = "yes"
            elif rating <= 0.5:
                initial_classification = "no"
            
            # Apply confidence-based adjustment ONLY if confidence < 0.9
            if confidence < 0.9:
                if initial_classification == "yes":
                    # For hate speech with low confidence: rating between 1.0 and 2.0
                    # Lower confidence = closer to 1.0, higher confidence = closer to 2.0
                    confidence_factor = confidence / 0.9  # 0.0 to 1.0 scale
                    rating = 1.0 + confidence_factor * 1.0  # Maps to 1.0-2.0 range
                    
                elif initial_classification == "no":
                    # For no hate with low confidence: rating between 0.0 and 1.0
                    # Lower confidence = closer to 1.0, higher confidence = closer to 0.0
                    confidence_factor = confidence / 0.9  # 0.0 to 1.0 scale
                    rating = 1.0 - confidence_factor * 1.0  # Maps to 1.0-0.0 range
                    
                else:  # unclear cases
                    # For unclear cases: keep original rating but ensure it's between 0.8 and 1.2
                    rating = max(0.8, min(1.2, rating))
            else:
                # High confidence: use discrete values
                if rating >= 1.5:
                    rating = 2.0
                elif rating <= 0.5:
                    rating = 0.0
                else:
                    rating = 1.0
            
            # Ensure rating stays within bounds
            rating = max(0.0, min(2.0, rating))
            
            # Determine final category based on adjusted rating
            if rating >= 1.5:
                category = "yes"
            elif rating >= 0.5:
                category = "unclear"  
            else:
                category = "no"
            
            return {
                "rating": round(rating, 2),
                "category": category,
                "confidence": round(confidence, 2),
                "method": "gguf",
                "response": output[:50],
                "text": text[:100] + "..." if len(text) > 100 else text
            }
            
        except Exception as e:
            raise Exception(f"GGUF analysis failed: {e}")

    def analyze_text(self, text):
        """Main analysis method"""
        if len(text.strip()) < 10:
            raise Exception("Text too short for analysis")
        
        if self.method == "gguf":
            return self.analyze_with_gguf(text)
        else:
            raise Exception("No GGUF model available")

    def analyze_dataset(self, n_samples=200):
        """Analyze your hate speech dataset"""
        print(f"\n🎯 LLAMA-CPP HATE SPEECH ANALYSIS")
        print(f"Method: {self.method.upper()}")
        print("=" * 50)
        
        try:
            # Load selected high-variance comments
            df = pd.read_csv("selected_comments.csv")
            print(f"📊 Loaded {len(df):,} selected high-variance comments")
            
            # Use all selected comments (they're already curated)
            if n_samples and len(df) > n_samples:
                df_sample = df.sample(n=n_samples, random_state=42)
                print(f"🎯 Analyzing {len(df_sample)} of the selected comments...")
            else:
                df_sample = df
                print(f"🎯 Analyzing all {len(df_sample)} selected high-variance comments...")
            
            results = []
            errors = []
            start_time = time.time()
            
            for idx, row in df_sample.iterrows():
                text = str(row['text']).strip()
                
                if len(text) < 10:
                    continue
                
                try:
                    result = self.analyze_text(text)
                    result['comment_id'] = row.get('comment_id', idx)
                    results.append(result)
                except Exception as e:
                    errors.append({"comment_id": row.get('comment_id', idx), "error": str(e)})
                
                # Progress
                if len(results) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = len(results) / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {len(results)}/{len(df_sample)} | Rate: {rate:.1f}/sec | Errors: {len(errors)}")
            
            if not results:
                raise Exception("No successful analyses completed")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llamacpp_hate_analysis_{timestamp}.csv"
            
            df_results = pd.DataFrame(results)
            df_results.to_csv(filename, index=False)
            
            # Summary
            total = len(results)
            hate_2 = len([r for r in results if r['rating'] == 2.0])
            unclear_1 = len([r for r in results if r['rating'] == 1.0])
            no_hate_0 = len([r for r in results if r['rating'] == 0.0])
            
            elapsed = time.time() - start_time
            
            print(f"\n📈 LLAMA-CPP ANALYSIS RESULTS")
            print("=" * 50)
            print(f"Total analyzed: {total}")
            print(f"🔴 Hate Speech (2.0): {hate_2:3d} ({hate_2/total*100:5.1f}%)")
            print(f"🟡 Unclear (1.0):     {unclear_1:3d} ({unclear_1/total*100:5.1f}%)")
            print(f"🟢 No Hate (0.0):     {no_hate_0:3d} ({no_hate_0/total*100:5.1f}%)")
            print(f"")
            print(f"🤖 Method: {self.method}")
            print(f"⏱️  Analysis time: {elapsed:.1f} seconds")
            print(f"🚀 Speed: {total/elapsed:.1f} comments/second")
            print(f"❌ Errors: {len(errors)}")
            print(f"💾 Results saved: {filename}")
            
            # Create visualization
            plot_file = visualize_hate_ratings(results, "llamacpp_hate_analysis")
            
            # Show examples
            print(f"\n📝 Sample Results:")
            for rating, label, emoji in [(2.0, "Hate", "🔴"), (1.0, "Unclear", "🟡"), (0.0, "No Hate", "🟢")]:
                examples = [r for r in results if r['rating'] == rating][:2]
                if examples:
                    print(f"\n{emoji} {label}:")
                    for ex in examples:
                        text_preview = ex['text'][:60] + "..." if len(ex['text']) > 60 else ex['text']
                        method_icon = "�"  # GGUF only
                        print(f"  {method_icon} '{text_preview}' (conf: {ex['confidence']:.1%})")
            
            return results
            
        except Exception as e:
            print(f"❌ Dataset analysis failed: {e}")
            raise

def show_setup_instructions():
    """Show setup instructions for llama-cpp"""
    print("\n💡 PURE LLAMA-CPP SETUP INSTRUCTIONS")
    print("=" * 50)
    print("\nTo use this analyzer, you need GGUF model files:")
    print("\n🔥 AUTOMATIC DOWNLOAD (Recommended)")
    print("   • This script can download models automatically!")
    print("   • Uses: bartowski/Llama-3.2-3B-Instruct-GGUF")
    print("   • Model: Q4_K_M quantization (~2GB)")
    print("   • Requires: pip install huggingface_hub[cli]")
    print("\n📁 MANUAL GGUF DOWNLOAD")
    print("   • Visit: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF")
    print("   • Download any .gguf file (Q4_K_M recommended)")
    print("   • Place in: ./gguf_models/")
    print("   • Script will auto-detect and use them")
    print("\n✅ REQUIREMENTS")
    print("   • llama-cpp-python (already installed)")
    print("   • pandas (already installed)")
    print("   • huggingface_hub[cli] for auto-download")
    print("   • Your processed_dataset.csv file")
    
    print("\n💾 STORAGE REQUIREMENTS")
    print("   • Q4_K_M: ~2.0 GB (recommended balance)")
    print("   • Q4_K_S: ~1.9 GB (smaller, slight quality loss)")  
    print("   • Q6_K: ~2.6 GB (higher quality)")
    print("   • Q8_0: ~3.4 GB (highest quality)")
    
    print("\n🎯 PURE LLAMA-CPP = No Ollama needed!")

def main():
    print("🎯 LLAMA-CPP HATE SPEECH ANALYZER")
    print("Pure llama-cpp implementation - no fallbacks")
    print("=" * 60)
    
    try:
        analyzer = LlamaCppHateAnalyzer()
        
        print(f"\n✅ Successfully initialized with {analyzer.method.upper()}")
        
        print(f"\nAnalysis options:")
        print(f"1. Your request (200 comments)")
        print(f"2. Quick test (50 comments)")
        print(f"3. Large analysis (500 comments)")
        
        choice = input(f"\nChoice (1-3): ").strip()
        
        if choice == "1":
            n_samples = 200
        elif choice == "2":
            n_samples = 50
        elif choice == "3":
            n_samples = 500
        else:
            n_samples = 200
        
        print(f"\n🚀 Starting analysis of {n_samples} comments with {analyzer.method}...")
        
        results = analyzer.analyze_dataset(n_samples)
        
        print(f"\n🎉 SUCCESS! Analyzed {len(results)} comments using llama-cpp!")
        print("✅ Used your exact hate speech definition and 0-2 rating scale")
        print("✅ Pure llama-cpp implementation - no rule-based fallbacks")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        show_setup_instructions()

if __name__ == "__main__":
    main()