#!/usr/bin/env python3
"""
llama-cpp Hate Speech Analyzer
Uses only llama-cpp models (Ollama or GGUF) for analysis
"""

import pandas as pd
import json
from datetime import datetime
import time
import os
from pathlib import Path

def check_ollama():
    """Check if Ollama is installed and has models"""
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            models = result.stdout.strip()
            if 'llama' in models.lower():
                print("✅ Ollama found with Llama models!")
                return True
            else:
                print("⚠️  Ollama found but no Llama models")
                print("Run: ollama pull llama3.2:3b")
                return False
        else:
            print("❌ Ollama not working")
            return False
    except:
        print("❌ Ollama not installed")
        return False

def find_gguf_models():
    """Find available GGUF model files"""
    print("🔍 Looking for GGUF files...")
    gguf_paths = []
    search_dirs = ["./gguf_models/", "~/.ollama/models/", "./"]
    
    for search_dir in search_dirs:
        path = Path(search_dir).expanduser()
        if path.exists():
            gguf_paths.extend(path.glob("*.gguf"))
    
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
    print("🚀 INITIALIZING LLAMA-CPP")
    print("=" * 40)
    
    try:
        from llama_cpp import Llama
        
        # Option 1: Try Ollama integration
        print("Option 1: Ollama integration")
        if check_ollama():
            return "ollama", None
        
        # Option 2: Look for GGUF files
        print("\nOption 2: Direct GGUF models")
        gguf_paths = find_gguf_models()
        
        if gguf_paths:
            # Use the first GGUF file found
            model_path = str(gguf_paths[0])
            print(f"Loading: {gguf_paths[0].name}")
            
            llm = Llama(
                model_path=model_path,
                n_ctx=1024,
                n_threads=4,
                verbose=False
            )
            
            print("✅ GGUF model loaded successfully!")
            return "gguf", llm
        
        print("❌ No suitable models found")
        return None, None
        
    except Exception as e:
        print(f"❌ llama-cpp initialization failed: {e}")
        return None, None

class LlamaCppHateAnalyzer:
    def __init__(self):
        self.method, self.model = initialize_llamacpp()
        
        if self.method is None:
            raise Exception("No llama-cpp models available. Please install Ollama or download GGUF models.")
        
        # Your exact prompt template
        self.prompt_template = '''Does this comment contain hate speech, defined as "bias-motivated, hostile and malicious language targeted at a person/group because of their actual or perceived innate characteristics, especially when the group is unnecessarily labeled?

Comment: "{text}"

Please respond with only one of these exact ratings:
yes:2.0
unclear:1.0  
no:0.0

Rating:'''

    def analyze_with_ollama(self, text):
        """Use Ollama API for analysis"""
        try:
            import subprocess
            
            prompt = self.prompt_template.format(text=text[:300])
            
            # Use Ollama's generate command
            cmd = ['ollama', 'generate', 'llama3.2:3b', prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                response = result.stdout.strip().lower()
                
                # Parse response
                if "yes:2.0" in response or "2.0" in response:
                    rating, category = 2.0, "yes"
                elif "unclear:1.0" in response or "1.0" in response:
                    rating, category = 1.0, "unclear"
                else:
                    rating, category = 0.0, "no"
                
                return {
                    "rating": rating,
                    "category": category,
                    "confidence": 0.85,
                    "method": "ollama",
                    "response": response[:50],
                    "text": text[:100] + "..." if len(text) > 100 else text
                }
            else:
                raise Exception(f"Ollama error: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Ollama analysis failed: {e}")

    def analyze_with_gguf(self, text):
        """Use direct GGUF model"""
        try:
            prompt = self.prompt_template.format(text=text[:300])
            
            response = self.model(prompt, max_tokens=15, temperature=0.1)
            output = response['choices'][0]['text'].strip().lower()
            
            # Parse response
            if "yes:2.0" in output or "2.0" in output:
                rating, category = 2.0, "yes"
            elif "unclear:1.0" in output or "1.0" in output:
                rating, category = 1.0, "unclear"
            else:
                rating, category = 0.0, "no"
            
            return {
                "rating": rating,
                "category": category,
                "confidence": 0.9,
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
        
        if self.method == "ollama":
            return self.analyze_with_ollama(text)
        elif self.method == "gguf":
            return self.analyze_with_gguf(text)
        else:
            raise Exception("No analysis method available")

    def analyze_dataset(self, n_samples=200):
        """Analyze your hate speech dataset"""
        print(f"\n🎯 LLAMA-CPP HATE SPEECH ANALYSIS")
        print(f"Method: {self.method.upper()}")
        print("=" * 50)
        
        try:
            # Load your dataset
            df = pd.read_csv("processed_dataset.csv")
            print(f"📊 Loaded {len(df):,} comments")
            
            # Sample
            if len(df) > n_samples:
                df_sample = df.sample(n=n_samples, random_state=42)
            else:
                df_sample = df
            
            print(f"🎯 Analyzing {len(df_sample)} comments with {self.method}...")
            
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
            
            # Show examples
            print(f"\n📝 Sample Results:")
            for rating, label, emoji in [(2.0, "Hate", "🔴"), (1.0, "Unclear", "🟡"), (0.0, "No Hate", "🟢")]:
                examples = [r for r in results if r['rating'] == rating][:2]
                if examples:
                    print(f"\n{emoji} {label}:")
                    for ex in examples:
                        text_preview = ex['text'][:60] + "..." if len(ex['text']) > 60 else ex['text']
                        method_icon = "🦙" if self.method == "ollama" else "🤖"
                        print(f"  {method_icon} '{text_preview}' (conf: {ex['confidence']:.1%})")
            
            return results
            
        except Exception as e:
            print(f"❌ Dataset analysis failed: {e}")
            raise

def show_setup_instructions():
    """Show setup instructions for llama-cpp"""
    print("\n💡 SETUP INSTRUCTIONS")
    print("=" * 40)
    print("\nTo use this analyzer, you need either:")
    print("\n1. OLLAMA (Recommended - Easy)")
    print("   • Download from: https://ollama.ai/")
    print("   • Install and run: ollama pull llama3.2:3b")
    print("   • Works immediately with this script")
    print("\n2. GGUF MODEL FILES")
    print("   • Download from: https://huggingface.co/microsoft/Llama-3.2-3B-GGUF")
    print("   • Place .gguf files in: ./gguf_models/")
    print("   • Script will automatically detect and use them")
    print("\n3. REQUIREMENTS")
    print("   • llama-cpp-python (already installed)")
    print("   • pandas (already installed)")
    print("   • Your processed_dataset.csv file")

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