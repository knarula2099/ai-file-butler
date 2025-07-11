#!/usr/bin/env python3
# demo/week2_demo.py

"""
AI File Butler Week 2 Demo - Advanced Features

Demonstrates:
1. LLM-powered semantic organization
2. ML clustering for pattern discovery
3. Advanced duplicate detection
4. Cost tracking and optimization
5. Comparative analysis of strategies
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.create_test_data import TestDataGenerator
from file_butler.core.scanner import FileScanner
from file_butler.core.extractor import FeatureExtractorManager
from file_butler.engines.rules import RuleBasedEngine
from file_butler.engines.clustering import MLClusteringEngine, ClusteringConfig
from file_butler.engines.llm import LLMOrganizationEngine, LLMConfig
from file_butler.detectors.duplicates import DuplicateDetector
from file_butler.utils.cost_tracker import CostTracker

class Week2Demo:
    """Comprehensive demo of Week 2 advanced features"""
    
    def __init__(self):
        self.generator = TestDataGenerator()
        self.scanner = FileScanner()
        self.extractor = FeatureExtractorManager()
        self.cost_tracker = CostTracker()
    
    def run_comprehensive_demo(self):
        """Run comprehensive Week 2 feature demonstration"""
        
        print("🚀 AI File Butler - Week 2 Advanced Features Demo")
        print("=" * 60)
        
        # Create test data
        test_path = self._setup_test_data()
        
        # Demo 1: Strategy Comparison
        print("\n🧪 DEMO 1: Strategy Comparison")
        print("-" * 40)
        self._demo_strategy_comparison(test_path)
        
        # Demo 2: LLM Deep Dive
        if self._check_llm_availability():
            print("\n🤖 DEMO 2: LLM Semantic Analysis Deep Dive")
            print("-" * 45)
            self._demo_llm_features(test_path)
            
            # Demo 2.5: Custom Prompt Examples
            print("\n🎯 DEMO 2.5: Custom Prompt Examples")
            print("-" * 35)
            self._demo_custom_prompts(test_path)
        
        # Demo 3: ML Clustering Analysis
        if self._check_ml_availability():
            print("\n📊 DEMO 3: ML Clustering Analysis")
            print("-" * 35)
            self._demo_ml_clustering(test_path)
        
        # Demo 4: Advanced Duplicate Detection
        print("\n🔍 DEMO 4: Advanced Duplicate Detection")
        print("-" * 40)
        self._demo_advanced_duplicates(test_path)
        
        # Demo 5: Cost Optimization
        if self.cost_tracker.total_cost_usd > 0:
            print("\n💰 DEMO 5: Cost Tracking & Optimization")
            print("-" * 40)
            self._demo_cost_tracking()
        
        print("\n🎉 Week 2 Demo Complete!")
        print("💡 Try the enhanced CLI: python -m file_butler.interfaces.enhanced_cli --help")
    
    def _setup_test_data(self):
        """Setup comprehensive test data"""
        print("📁 Creating comprehensive test dataset...")
        
        # Create all scenarios for thorough testing
        scenarios = [
            self.generator.create_messy_downloads_scenario(),
            self.generator.create_document_archive_scenario(),
            self.generator.create_photo_collection_scenario()
        ]
        
        # Use the messy downloads as primary test
        test_path = scenarios[0]
        print(f"✅ Test data ready: {test_path}")
        return test_path
    
    def _demo_strategy_comparison(self, test_path):
        """Compare different organization strategies"""
        
        # Scan files once
        scan_result = self.scanner.scan_directory(test_path)
        files_with_features = self.extractor.extract_batch(scan_result.files)
        
        print(f"📊 Comparing strategies on {len(files_with_features)} files...")
        
        strategies = {
            'Rules': RuleBasedEngine(),
            'ML Clustering': self._get_ml_engine() if self._check_ml_availability() else None,
            'LLM Semantic': self._get_llm_engine() if self._check_llm_availability() else None
        }
        
        results = {}
        
        for name, engine in strategies.items():
            if engine is None:
                print(f"⏭️  Skipping {name} (dependencies not available)")
                continue
            
            print(f"\n🔬 Testing: {name}")
            
            try:
                start_time = time.time()
                plan = engine.organize(files_with_features)
                execution_time = time.time() - start_time
                
                # Analyze plan
                summary = plan.summary()
                
                results[name] = {
                    'time': execution_time,
                    'total_actions': summary['total_actions'],
                    'directories': summary['directories_to_create'],
                    'moves': summary['files_to_move'],
                    'deletes': summary['files_to_delete']
                }
                
                print(f"  ⏱️  Execution time: {execution_time:.2f} seconds")
                print(f"  📊 Actions generated: {summary['total_actions']}")
                print(f"  📁 Directories created: {summary['directories_to_create']}")
                print(f"  📦 Files to move: {summary['files_to_move']}")
                
                # Show sample directory names
                create_actions = [a for a in plan.actions if a.action_type.value == 'create_dir']
                if create_actions:
                    print(f"  📂 Sample directories:")
                    for action in create_actions[:3]:
                        dir_name = Path(action.destination).name
                        print(f"    • {dir_name}")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
        
        # Comparison summary
        if len(results) > 1:
            print(f"\n📊 Strategy Comparison Summary:")
            print(f"{'Strategy':<15} {'Time (s)':<10} {'Actions':<10} {'Dirs':<8}")
            print("-" * 50)
            
            for name, data in results.items():
                print(f"{name:<15} {data['time']:<10.2f} {data['total_actions']:<10} {data['directories']:<8}")
    
    def _demo_llm_features(self, test_path):
        """Demonstrate LLM semantic analysis features"""
        
        if not self._check_llm_availability():
            print("⏭️  LLM features not available")
            return
        
        # Scan and prepare files
        scan_result = self.scanner.scan_directory(test_path)
        # Limit files for cost control in demo
        sample_files = self.extractor.extract_batch(scan_result.files[:8])
        
        print(f"🤖 Analyzing {len(sample_files)} files with LLM...")
        
        try:
            # Initialize LLM engine with cost controls
            config = LLMConfig(
                provider='openai',
                model='gpt-3.5-turbo',
                cost_limit_usd=2.0,  # Low limit for demo
                max_file_size_kb=20
            )
            engine = LLMOrganizationEngine(config)
            
            # Step 1: File content analysis
            print("\n📖 Step 1: Content Analysis")
            file_summaries = engine._analyze_file_contents(sample_files)
            
            categories = {}
            for summary in file_summaries:
                category = summary.get('category', 'Unknown')
                if category not in categories:
                    categories[category] = []
                categories[category].append(summary)
            
            print(f"  🏷️  Detected {len(categories)} semantic categories:")
            for category, files in categories.items():
                print(f"    📂 {category}: {len(files)} files")
                
                # Show detailed analysis for first file in each category
                if files:
                    sample = files[0]
                    print(f"      📄 Example: {sample['file_name']}")
                    print(f"         Summary: {sample.get('summary', 'N/A')[:60]}...")
                    print(f"         Topics: {', '.join(sample.get('topics', [])[:3])}")
                    print(f"         Confidence: {sample.get('confidence', 0):.2f}")
            
            # Step 2: Folder structure generation
            print(f"\n🏗️  Step 2: Intelligent Folder Structure")
            folder_structure = engine._generate_folder_structure(file_summaries)
            
            print(f"  📁 Proposed hierarchy:")
            for directory in folder_structure.get('directories', []):
                print(f"    📂 {directory}")
            
            if folder_structure.get('reasoning'):
                print(f"\n  💭 LLM Reasoning:")
                print(f"     {folder_structure['reasoning']}")
            
            # Step 3: File mapping
            print(f"\n🎯 Step 3: Semantic File Mapping")
            file_mappings = engine._map_files_to_folders(file_summaries, folder_structure)
            
            for mapping in file_mappings[:5]:
                source_name = Path(mapping['source_path']).name
                target_folder = Path(mapping['target_path']).parent.name
                print(f"    📄 {source_name} → 📂 {target_folder}")
                print(f"       Reason: {mapping['reason'][:50]}...")
            
            # Cost tracking
            total_cost = engine.cost_tracker.total_cost_usd
            print(f"\n💰 LLM Usage Cost: ${total_cost:.4f}")
            
        except Exception as e:
            print(f"❌ LLM demo failed: {e}")
            if "API key" in str(e).lower():
                print("💡 Set OPENAI_API_KEY environment variable to enable LLM features")
    
    def _demo_ml_clustering(self, test_path):
        """Demonstrate ML clustering analysis"""
        
        if not self._check_ml_availability():
            print("⏭️  ML clustering not available")
            return
        
        # Scan files
        scan_result = self.scanner.scan_directory(test_path)
        files_with_features = self.extractor.extract_batch(scan_result.files)
        
        print(f"📊 Performing ML clustering on {len(files_with_features)} files...")
        
        try:
            # Test different clustering algorithms
            algorithms = ['dbscan', 'kmeans', 'hierarchical']
            
            for algorithm in algorithms:
                print(f"\n🔬 Testing {algorithm.upper()} clustering:")
                
                config = ClusteringConfig(
                    algorithm=algorithm,
                    min_samples=2,
                    eps=0.5,
                    n_clusters=5 if algorithm == 'kmeans' else None
                )
                
                engine = MLClusteringEngine(config)
                
                # Get clustering analysis
                start_time = time.time()
                viz_data = engine.get_cluster_visualization_data(files_with_features)
                clustering_time = time.time() - start_time
                
                if 'error' in viz_data:
                    print(f"  ❌ {viz_data['error']}")
                    continue
                
                print(f"  ⏱️  Clustering time: {clustering_time:.2f} seconds")
                print(f"  🎯 Clusters found: {viz_data['n_clusters']}")
                print(f"  🔍 Noise points: {viz_data['n_noise']}")
                
                # Analyze cluster distribution
                cluster_sizes = {}
                for point in viz_data['points']:
                    cluster_id = point['cluster']
                    cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
                
                print(f"  📊 Cluster distribution:")
                for cluster_id, size in sorted(cluster_sizes.items()):
                    if cluster_id == -1:
                        print(f"    🔍 Noise: {size} files")
                    else:
                        print(f"    📂 Cluster {cluster_id}: {size} files")
                
                # Show sample files from largest cluster
                if viz_data['n_clusters'] > 0:
                    largest_cluster = max([k for k in cluster_sizes.keys() if k != -1], 
                                        key=lambda k: cluster_sizes[k])
                    cluster_files = [p for p in viz_data['points'] if p['cluster'] == largest_cluster]
                    
                    print(f"  📄 Sample from largest cluster ({largest_cluster}):")
                    for file_point in cluster_files[:3]:
                        print(f"    • {file_point['file_name']} ({file_point['file_type']})")
                
                # Explained variance for dimensionality reduction
                if viz_data.get('explained_variance'):
                    total_variance = sum(viz_data['explained_variance'])
                    print(f"  📈 Explained variance: {total_variance:.2f}")
            
        except Exception as e:
            print(f"❌ ML clustering demo failed: {e}")
    
    def _demo_advanced_duplicates(self, test_path):
        """Demonstrate advanced duplicate detection"""
        
        # Scan files
        scan_result = self.scanner.scan_directory(test_path)
        files_with_features = self.extractor.extract_batch(scan_result.files)
        
        print(f"🔍 Advanced duplicate analysis on {len(files_with_features)} files...")
        
        # 1. Exact duplicates
        print("\n🎯 Exact Duplicate Detection:")
        detector = DuplicateDetector()
        duplicate_groups = detector.find_exact_duplicates(files_with_features)
        
        if duplicate_groups:
            stats = detector.get_duplicate_statistics(duplicate_groups)
            print(f"  📊 Found {stats['duplicate_groups']} groups of exact duplicates")
            print(f"  🔄 Total duplicate files: {stats['total_duplicates']}")
            print(f"  💾 Wasted space: {stats['wasted_space_mb']:.2f} MB")
            print(f"  📈 Space savings potential: {stats['space_savings_percent']:.1f}%")
            
            # Show sample duplicate groups
            for i, group in enumerate(duplicate_groups[:2]):
                print(f"\n  📂 Group {i+1}: {len(group)} identical files")
                for file_info in group:
                    print(f"    📄 {file_info.name} ({file_info.size_mb:.2f} MB)")
        else:
            print("  ✅ No exact duplicates found")
        
        # 2. Image similarity detection
        image_files = [f for f in files_with_features if f.file_type.value == 'image']
        if image_files:
            print(f"\n🖼️  Image Similarity Detection:")
            print(f"  📊 Analyzing {len(image_files)} images...")
            
            try:
                from file_butler.engines.advanced_features import ImageHashExtractor
                
                extractor = ImageHashExtractor()
                similar_groups = extractor.find_similar_images(image_files, threshold=5)
                
                if similar_groups:
                    print(f"  🎯 Found {len(similar_groups)} groups of similar images")
                    
                    for i, group in enumerate(similar_groups[:2]):
                        print(f"\n    📸 Similar Group {i+1}: {len(group)} images")
                        for file_info in group:
                            print(f"      📄 {file_info.name}")
                else:
                    print("  ✅ No similar images found")
                    
            except ImportError:
                print("  ⏭️  Image similarity requires: pip install imagehash Pillow")
        else:
            print("\n🖼️  No images found for similarity analysis")
        
        # 3. Text similarity detection
        text_files = [f for f in files_with_features if f.file_type.value in ['document', 'code']]
        if text_files and len(text_files) > 1:
            print(f"\n📄 Text Similarity Detection:")
            print(f"  📊 Analyzing {len(text_files)} text files...")
            
            try:
                from file_butler.detectors.duplicates import NearDuplicateDetector
                
                near_detector = NearDuplicateDetector(similarity_threshold=0.7)
                similar_text_groups = near_detector.find_similar_text_files(text_files)
                
                if similar_text_groups:
                    print(f"  🎯 Found {len(similar_text_groups)} groups of similar text files")
                    
                    for i, group in enumerate(similar_text_groups[:2]):
                        print(f"\n    📝 Similar Group {i+1}: {len(group)} files")
                        for file_info in group:
                            print(f"      📄 {file_info.name}")
                else:
                    print("  ✅ No similar text files found")
                    
            except Exception as e:
                print(f"  ⚠️  Text similarity analysis failed: {e}")
        else:
            print("\n📄 Not enough text files for similarity analysis")
    
    def _demo_custom_prompts(self, test_path):
        """Demonstrate custom prompt functionality"""
        
        if not self._check_llm_availability():
            print("⏭️  LLM features not available")
            return
        
        # Scan and prepare files
        scan_result = self.scanner.scan_directory(test_path)
        # Limit files for cost control in demo
        sample_files = self.extractor.extract_batch(scan_result.files[:6])
        
        print(f"🎯 Testing custom prompts on {len(sample_files)} files...")
        
        # Example custom prompts
        custom_prompts = [
            {
                "name": "Organize by Year",
                "prompt": "Organize these files by the year they were created or modified. Create folders like '2024', '2023', '2022', etc. Look for dates in file names, metadata, or content.",
                "description": "Groups files by creation/modification year"
            },
            {
                "name": "Organize by Location", 
                "prompt": "Organize these files by geographic location. Extract location information from file names, metadata, or content. Create folders like 'New York', 'London', 'Tokyo', etc.",
                "description": "Groups files by geographic location"
            },
            {
                "name": "Organize by Project",
                "prompt": "Organize these files by project name. Look for project identifiers in file names or content. Create folders for each unique project.",
                "description": "Groups files by project name"
            }
        ]
        
        for i, prompt_info in enumerate(custom_prompts, 1):
            print(f"\n🔬 Test {i}: {prompt_info['name']}")
            print(f"   📝 {prompt_info['description']}")
            
            try:
                # Initialize LLM engine with custom prompt
                config = LLMConfig(
                    provider='openai',
                    model='gpt-3.5-turbo',
                    cost_limit_usd=1.0,  # Low limit for demo
                    max_file_size_kb=20,
                    custom_prompt=prompt_info['prompt']
                )
                engine = LLMOrganizationEngine(config)
                
                # Analyze files with custom prompt
                file_summaries = engine._analyze_file_contents(sample_files)
                
                # Show results
                categories = {}
                for summary in file_summaries:
                    category = summary.get('category', 'Unknown')
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(summary)
                
                print(f"   🏷️  Detected categories:")
                for category, files in categories.items():
                    print(f"      📂 {category}: {len(files)} files")
                
                # Show folder structure
                folder_structure = engine._generate_folder_structure(file_summaries)
                print(f"   📁 Proposed folders:")
                for directory in folder_structure.get('directories', []):
                    print(f"      📂 {directory}")
                
                # Show cost
                total_cost = engine.cost_tracker.total_cost_usd
                print(f"   💰 Cost: ${total_cost:.4f}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        print(f"\n💡 Custom Prompt Tips:")
        print("   • Be specific about what information to extract")
        print("   • Mention the desired folder structure")
        print("   • Include examples of expected folder names")
        print("   • Use the CLI: file-butler organize-with-prompt /path --prompt \"your prompt\"")
    
    def _demo_cost_tracking(self):
        """Demonstrate cost tracking and optimization"""
        
        print("💰 LLM Cost Analysis:")
        
        # Generate detailed report
        report = self.cost_tracker.generate_report()
        print(report)
        
        # Show optimization tips
        print("\n💡 Cost Optimization Tips:")
        print("  🎯 Use caching to avoid re-analyzing same files")
        print("  📏 Set file size limits to avoid processing large files")
        print("  🔄 Use batch processing to reduce API calls")
        print("  ⚡ Consider local models for privacy and cost savings")
        print("  📊 Monitor daily spending with cost limits")
        
        # Show daily breakdown if available
        daily_costs = self.cost_tracker.get_daily_costs(7)
        if daily_costs:
            print(f"\n📅 Recent Daily Usage:")
            for date, cost in sorted(daily_costs.items())[-5:]:
                print(f"  {date}: ${cost:.4f}")
    
    def _check_llm_availability(self):
        """Check if LLM features are available"""
        try:
            import openai
            return bool(os.getenv('OPENAI_API_KEY'))
        except ImportError:
            return False
    
    def _check_ml_availability(self):
        """Check if ML features are available"""
        try:
            import sklearn
            import pandas
            import numpy
            return True
        except ImportError:
            return False
    
    def _get_llm_engine(self):
        """Get configured LLM engine"""
        if not self._check_llm_availability():
            return None
        
        try:
            config = LLMConfig(
                provider='openai',
                model='gpt-3.5-turbo',
                cost_limit_usd=1.0,  # Low limit for demo
                max_file_size_kb=20
            )
            return LLMOrganizationEngine(config)
        except Exception:
            return None
    
    def _get_ml_engine(self):
        """Get configured ML engine"""
        if not self._check_ml_availability():
            return None
        
        try:
            config = ClusteringConfig(
                algorithm='dbscan',
                min_samples=2,
                eps=0.5
            )
            return MLClusteringEngine(config)
        except Exception:
            return None
    
    def interactive_demo(self):
        """Run interactive demo with user choices"""
        
        print("🎮 Interactive AI File Butler Demo")
        print("=" * 40)
        
        # Check available features
        llm_available = self._check_llm_availability()
        ml_available = self._check_ml_availability()
        
        print("\n🔧 Available Features:")
        print(f"  📏 Rule-based organization: ✅")
        print(f"  📊 ML clustering: {'✅' if ml_available else '❌ (pip install scikit-learn pandas numpy)'}")
        print(f"  🤖 LLM semantic analysis: {'✅' if llm_available else '❌ (set OPENAI_API_KEY)'}")
        
        # Create test data
        test_path = self._setup_test_data()
        
        while True:
            print("\n🎯 Choose a demo:")
            print("  1. Compare all organization strategies")
            print("  2. Deep dive into LLM analysis" + (" (requires API key)" if not llm_available else ""))
            print("  3. Explore ML clustering patterns" + (" (requires ML libraries)" if not ml_available else ""))
            print("  4. Advanced duplicate detection")
            print("  5. Custom prompt examples" + (" (requires API key)" if not llm_available else ""))
            print("  6. Cost tracking analysis" + (" (requires LLM usage)" if self.cost_tracker.total_cost_usd == 0 else ""))
            print("  7. Run full comprehensive demo")
            print("  0. Exit")
            
            try:
                choice = input("\nEnter your choice (0-6): ").strip()
                
                if choice == '0':
                    print("👋 Thanks for exploring AI File Butler!")
                    break
                elif choice == '1':
                    self._demo_strategy_comparison(test_path)
                elif choice == '2':
                    if llm_available:
                        self._demo_llm_features(test_path)
                    else:
                        print("❌ LLM features not available. Set OPENAI_API_KEY environment variable.")
                elif choice == '3':
                    if ml_available:
                        self._demo_ml_clustering(test_path)
                    else:
                        print("❌ ML features not available. Install: pip install scikit-learn pandas numpy")
                elif choice == '4':
                    self._demo_advanced_duplicates(test_path)
                elif choice == '5':
                    if llm_available:
                        self._demo_custom_prompts(test_path)
                    else:
                        print("❌ LLM features not available. Set OPENAI_API_KEY environment variable.")
                elif choice == '6':
                    if self.cost_tracker.total_cost_usd > 0:
                        self._demo_cost_tracking()
                    else:
                        print("❌ No cost data available. Run LLM analysis first.")
                elif choice == '7':
                    self.run_comprehensive_demo()
                else:
                    print("❌ Invalid choice. Please enter 0-7.")
                
                if choice != '0':
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main demo function with user choice"""
    
    print("🚀 AI File Butler - Week 2 Advanced Features")
    print("=" * 50)
    
    demo = Week2Demo()
    
    print("\nChoose demo mode:")
    print("1. 🎮 Interactive Demo (choose specific features)")
    print("2. 📋 Comprehensive Demo (show everything)")
    print("3. ⚡ Quick Test (basic functionality)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                demo.interactive_demo()
                break
            elif choice == '2':
                demo.run_comprehensive_demo()
                break
            elif choice == '3':
                # Quick test with rules engine
                test_path = demo._setup_test_data()
                print("\n⚡ Quick Test - Rules Engine:")
                demo._demo_strategy_comparison(test_path)
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Demo cancelled. Goodbye!")
            break

if __name__ == "__main__":
    main()