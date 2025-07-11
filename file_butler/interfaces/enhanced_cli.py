# file_butler/interfaces/enhanced_cli.py
import click
import os
import json
from pathlib import Path
from typing import Optional

from ..core.scanner import FileScanner
from ..core.extractor import FeatureExtractorManager
from ..core.executor import ActionExecutor
from ..engines.rules import RuleBasedEngine
from ..engines.clustering import MLClusteringEngine, ClusteringConfig
from ..engines.llm import LLMOrganizationEngine, LLMConfig
from ..detectors.duplicates import DuplicateDetector
from ..utils.config import get_config
from ..utils.cost_tracker import CostTracker

class EnhancedFileButlerCLI:
    """Enhanced CLI with ML and LLM capabilities"""
    
    def __init__(self):
        self.config = get_config()
        self.scanner = FileScanner()
        self.extractor = FeatureExtractorManager()
        self.executor = ActionExecutor()
        self.cost_tracker = CostTracker()
    
    def organize_directory(self, 
                          source_path: str,
                          target_path: Optional[str] = None,
                          strategy: str = "rules",
                          dry_run: bool = True,
                          find_duplicates: bool = True,
                          verbose: bool = False,
                          **kwargs):
        """Enhanced organization workflow with multiple strategies"""
        
        # Setup paths
        source_path = Path(source_path).resolve()
        if target_path is None:
            target_path = source_path / "organized"
        else:
            target_path = Path(target_path).resolve()
        
        if verbose:
            click.echo(f"🤖 AI File Butler - Enhanced Mode")
            click.echo(f"📁 Source: {source_path}")
            click.echo(f"📂 Target: {target_path}")
            click.echo(f"🧠 Strategy: {strategy}")
            click.echo(f"🔄 Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
            click.echo("-" * 60)
        
        # Step 1: Scan directory
        if verbose:
            click.echo("🔍 Scanning directory...")
        
        scan_result = self.scanner.scan_directory(str(source_path))
        
        if verbose:
            click.echo(f"✅ Scan complete: {len(scan_result.files)} files found")
        
        if not scan_result.files:
            click.echo("❌ No files found to organize!")
            return
        
        # Step 2: Extract features
        if verbose:
            click.echo("🔬 Extracting features...")
        
        files_with_features = self.extractor.extract_batch(scan_result.files)
        
        # Step 3: Choose and initialize organization engine
        if verbose:
            click.echo(f"🧠 Initializing {strategy} engine...")
        
        engine = self._create_organization_engine(strategy, **kwargs)
        
        if engine is None:
            click.echo(f"❌ Failed to initialize {strategy} engine")
            return
        
        # Step 4: Find duplicates (optional)
        duplicate_groups = []
        if find_duplicates:
            if verbose:
                click.echo("🔍 Finding duplicates...")
            
            detector = DuplicateDetector()
            duplicate_groups = detector.find_exact_duplicates(files_with_features)
            
            if verbose and duplicate_groups:
                click.echo(f"🔄 Found {len(duplicate_groups)} groups of duplicates")
        
        # Step 5: Generate organization plan
        if verbose:
            click.echo("📋 Generating organization plan...")
        
        plan = engine.organize(files_with_features)
        plan.source_directory = str(source_path)
        plan.target_directory = str(target_path)
        
        # Add duplicate deletion suggestions
        if duplicate_groups:
            detector = DuplicateDetector()
            deletion_actions = detector.suggest_deletions(duplicate_groups)
            plan.actions.extend(deletion_actions)
        
        # Step 6: Display plan summary
        self._display_enhanced_plan_summary(plan, strategy, verbose)
        
        # Step 7: Show cost information for LLM
        if strategy == "llm":
            self._display_cost_info()
        
        # Step 8: Execute plan
        if not dry_run:
            if click.confirm("Proceed with execution?"):
                self._execute_plan(plan, verbose)
            else:
                click.echo("❌ Execution cancelled")
        else:
            click.echo("\n💡 Run with --execute to apply changes")
        
        return plan
    
    def _create_organization_engine(self, strategy: str, **kwargs):
        """Create the appropriate organization engine"""
        
        if strategy == "rules":
            return RuleBasedEngine()
        
        elif strategy == "clustering" or strategy == "ml":
            try:
                config = ClusteringConfig(
                    algorithm=kwargs.get('clustering_algorithm', 'dbscan'),
                    min_samples=kwargs.get('min_samples', 2),
                    eps=kwargs.get('eps', 0.5),
                    n_clusters=kwargs.get('n_clusters'),
                    max_features=kwargs.get('max_features', 1000)
                )
                return MLClusteringEngine(config)
            except ImportError as e:
                click.echo(f"❌ ML clustering not available: {e}")
                click.echo("💡 Install with: pip install scikit-learn pandas numpy")
                return None
        
        elif strategy == "llm":
            try:
                config = LLMConfig(
                    provider=kwargs.get('llm_provider', 'openai'),
                    model=kwargs.get('llm_model', 'gpt-3.5-turbo'),
                    max_tokens=kwargs.get('max_tokens', 500),
                    temperature=kwargs.get('temperature', 0.2),
                    cost_limit_usd=kwargs.get('cost_limit', 10.0)
                )
                return LLMOrganizationEngine(config)
            except Exception as e:
                click.echo(f"❌ LLM engine not available: {e}")
                click.echo("💡 Check API keys and install: pip install openai")
                return None
        
        else:
            click.echo(f"❌ Unknown strategy: {strategy}")
            return None
    
    def _display_enhanced_plan_summary(self, plan, strategy, verbose=True):
        """Display enhanced organization plan summary"""
        
        summary = plan.summary()
        
        if verbose:
            click.echo(f"\n📋 Organization Plan Summary ({strategy.upper()}):")
            click.echo(f"  📁 Directories to create: {summary['directories_to_create']}")
            click.echo(f"  📦 Files to move: {summary['files_to_move']}")
            click.echo(f"  🗑️  Files to delete: {summary['files_to_delete']}")
            click.echo(f"  📊 Total actions: {summary['total_actions']}")
        
        # Strategy-specific information
        if strategy == "clustering":
            self._display_clustering_info(plan)
        elif strategy == "llm":
            self._display_llm_info(plan)
        
        # Show sample actions
        if verbose and plan.actions:
            click.echo("\n📄 Sample actions:")
            for i, action in enumerate(plan.actions[:8]):
                confidence_indicator = ""
                if hasattr(action, 'confidence') and action.confidence:
                    confidence_indicator = f" (confidence: {action.confidence:.1f})"
                click.echo(f"  {i+1}. {action}{confidence_indicator}")
            
            if len(plan.actions) > 8:
                click.echo(f"  ... and {len(plan.actions) - 8} more actions")
    
    def _display_clustering_info(self, plan):
        """Display clustering-specific information"""
        
        # Count cluster directories
        cluster_dirs = [a for a in plan.actions 
                       if a.action_type.value == 'create_dir' and 'Clustered' in a.destination]
        
        if cluster_dirs:
            click.echo(f"\n🎯 ML Clustering Results:")
            click.echo(f"  📊 Discovered clusters: {len(cluster_dirs)}")
            
            # Show cluster names
            cluster_names = [Path(a.destination).name for a in cluster_dirs[:5]]
            for name in cluster_names:
                click.echo(f"    📂 {name}")
            
            if len(cluster_dirs) > 5:
                click.echo(f"    ... and {len(cluster_dirs) - 5} more clusters")
    
    def _display_llm_info(self, plan):
        """Display LLM-specific information"""
        
        # Count LLM-generated directories
        llm_dirs = [a for a in plan.actions 
                   if a.action_type.value == 'create_dir' and a.reason and 'LLM' in a.reason]
        
        if llm_dirs:
            click.echo(f"\n🤖 LLM Analysis Results:")
            click.echo(f"  📊 Semantic categories: {len(llm_dirs)}")
            
            # Show semantic folder names
            for action in llm_dirs[:5]:
                folder_name = Path(action.destination).name
                click.echo(f"    📂 {folder_name}")
            
            if len(llm_dirs) > 5:
                click.echo(f"    ... and {len(llm_dirs) - 5} more categories")
    
    def _display_cost_info(self):
        """Display cost information for LLM usage"""
        
        total_cost = self.cost_tracker.total_cost_usd
        
        if total_cost > 0:
            click.echo(f"\n💰 LLM Usage Costs:")
            click.echo(f"  💵 Total spent: ${total_cost:.4f}")
            
            # Recent usage
            daily_costs = self.cost_tracker.get_daily_costs(7)
            if daily_costs:
                recent_cost = sum(daily_costs.values())
                click.echo(f"  📅 Last 7 days: ${recent_cost:.4f}")
            
            # Model breakdown
            breakdown = self.cost_tracker.get_model_breakdown()
            for provider, models in breakdown.items():
                click.echo(f"  🔧 {provider.title()}:")
                for model, stats in models.items():
                    click.echo(f"    {model}: ${stats['cost']:.4f} ({stats['calls']} calls)")
    
    def _execute_plan(self, plan, verbose=True):
        """Execute the organization plan with enhanced reporting"""
        
        self.executor.dry_run = False
        
        def progress_callback(current, total, action_desc):
            if verbose:
                progress = (current / total) * 100
                click.echo(f"\r⏳ Progress: {progress:.1f}% ({current}/{total})", nl=False)
        
        if verbose:
            click.echo("\n🚀 Executing plan...")
        
        result = self.executor.execute_plan(plan)
        
        if verbose:
            click.echo(f"\n✅ Execution complete!")
            click.echo(f"  ✓ Actions executed: {result.executed_actions}")
            if result.failed_actions > 0:
                click.echo(f"  ❌ Actions failed: {result.failed_actions}")
            click.echo(f"  📈 Success rate: {result.success_rate:.1f}%")
            click.echo(f"  ⏱️  Time: {result.execution_time:.2f} seconds")
        
        return result

# Enhanced CLI Commands
@click.group()
def cli():
    """AI File Butler - Enhanced intelligent file organization"""
    pass

@cli.command()
@click.argument('source_path', type=click.Path(exists=True))
@click.option('--target', '-t', type=click.Path(), help='Target directory for organized files')
@click.option('--strategy', '-s', 
              type=click.Choice(['rules', 'clustering', 'ml', 'llm']), 
              default='rules',
              help='Organization strategy to use')
@click.option('--execute', is_flag=True, help='Execute the plan (default is dry run)')
@click.option('--no-duplicates', is_flag=True, help='Skip duplicate detection')
@click.option('--quiet', '-q', is_flag=True, help='Minimal output')
# LLM-specific options
@click.option('--llm-provider', type=click.Choice(['openai', 'anthropic', 'local']), 
              default='openai', help='LLM provider for semantic analysis')
@click.option('--llm-model', default='gpt-3.5-turbo', help='LLM model to use')
@click.option('--cost-limit', type=float, default=10.0, help='Maximum cost limit for LLM usage ($)')
# ML-specific options
@click.option('--clustering-algorithm', type=click.Choice(['dbscan', 'kmeans', 'hierarchical']),
              default='dbscan', help='Clustering algorithm')
@click.option('--min-samples', type=int, default=2, help='Minimum samples for DBSCAN')
@click.option('--eps', type=float, default=0.5, help='Epsilon parameter for DBSCAN')
@click.option('--n-clusters', type=int, help='Number of clusters (for k-means)')
def organize(source_path, target, strategy, execute, no_duplicates, quiet, 
            llm_provider, llm_model, cost_limit,
            clustering_algorithm, min_samples, eps, n_clusters):
    """Organize files using advanced AI strategies"""
    
    butler = EnhancedFileButlerCLI()
    
    try:
        # Prepare strategy-specific parameters
        strategy_params = {}
        
        if strategy == 'llm':
            strategy_params.update({
                'llm_provider': llm_provider,
                'llm_model': llm_model,
                'cost_limit': cost_limit
            })
        elif strategy in ['clustering', 'ml']:
            strategy_params.update({
                'clustering_algorithm': clustering_algorithm,
                'min_samples': min_samples,
                'eps': eps,
                'n_clusters': n_clusters
            })
        
        plan = butler.organize_directory(
            source_path=source_path,
            target_path=target,
            strategy=strategy,
            dry_run=not execute,
            find_duplicates=not no_duplicates,
            verbose=not quiet,
            **strategy_params
        )
        
        if not quiet:
            click.echo(f"\n🎉 File organization complete using {strategy} strategy!")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if not quiet:
            import traceback
            traceback.print_exc()
        raise click.Abort()

@cli.command()
@click.argument('source_path', type=click.Path(exists=True))
@click.option('--strategy', type=click.Choice(['clustering', 'ml']), default='clustering')
@click.option('--algorithm', type=click.Choice(['dbscan', 'kmeans', 'hierarchical']), default='dbscan')
@click.option('--show-features', is_flag=True, help='Show extracted features')
def analyze(source_path, strategy, algorithm, show_features):
    """Analyze files using ML clustering without organizing"""
    
    click.echo("🔬 Analyzing file patterns with ML...")
    
    try:
        from ..engines.clustering import MLClusteringEngine, ClusteringConfig
        
        # Setup
        scanner = FileScanner()
        extractor = FeatureExtractorManager()
        
        # Scan and extract features
        scan_result = scanner.scan_directory(source_path)
        files_with_features = extractor.extract_batch(scan_result.files)
        
        click.echo(f"📁 Analyzed {len(files_with_features)} files")
        
        # Initialize clustering engine
        config = ClusteringConfig(algorithm=algorithm)
        engine = MLClusteringEngine(config)
        
        # Get visualization data
        viz_data = engine.get_cluster_visualization_data(files_with_features)
        
        if 'error' in viz_data:
            click.echo(f"❌ Analysis failed: {viz_data['error']}")
            return
        
        # Display results
        click.echo(f"\n📊 Clustering Results ({algorithm.upper()}):")
        click.echo(f"  🎯 Clusters found: {viz_data['n_clusters']}")
        click.echo(f"  🔍 Noise points: {viz_data['n_noise']}")
        
        # Show cluster distribution
        cluster_counts = {}
        for point in viz_data['points']:
            cluster_id = point['cluster']
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        click.echo(f"\n📈 Cluster Distribution:")
        for cluster_id, count in sorted(cluster_counts.items()):
            if cluster_id == -1:
                click.echo(f"  🔍 Noise: {count} files")
            else:
                click.echo(f"  📂 Cluster {cluster_id}: {count} files")
        
        # Show sample files from each cluster
        click.echo(f"\n📄 Sample Files by Cluster:")
        for cluster_id in sorted(set(p['cluster'] for p in viz_data['points'])):
            if cluster_id == -1:
                continue
            
            cluster_files = [p for p in viz_data['points'] if p['cluster'] == cluster_id]
            click.echo(f"\n  📂 Cluster {cluster_id} ({len(cluster_files)} files):")
            
            for file_point in cluster_files[:3]:
                click.echo(f"    📄 {file_point['file_name']} ({file_point['file_type']})")
            
            if len(cluster_files) > 3:
                click.echo(f"    ... and {len(cluster_files) - 3} more files")
        
        if show_features:
            # Display feature information
            click.echo(f"\n🔬 Feature Extraction Details:")
            click.echo(f"  📊 Explained variance: {viz_data.get('explained_variance', 'N/A')}")
        
    except ImportError:
        click.echo("❌ ML analysis requires: pip install scikit-learn pandas numpy")
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}")

@cli.command()
@click.argument('source_path', type=click.Path(exists=True))
@click.option('--provider', type=click.Choice(['openai', 'anthropic', 'local']), default='openai')
@click.option('--model', default='gpt-3.5-turbo')
@click.option('--max-files', type=int, default=10, help='Maximum files to analyze (cost control)')
@click.option('--show-analysis', is_flag=True, help='Show detailed LLM analysis')
def preview_llm(source_path, provider, model, max_files, show_analysis):
    """Preview LLM analysis without organizing"""
    
    click.echo("🤖 Previewing LLM semantic analysis...")
    
    try:
        from ..engines.llm import LLMOrganizationEngine, LLMConfig
        
        # Check API key
        if provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
            click.echo("❌ OPENAI_API_KEY environment variable not set")
            click.echo("💡 Set it with: export OPENAI_API_KEY=your_key_here")
            return
        
        # Setup
        scanner = FileScanner()
        extractor = FeatureExtractorManager()
        
        # Scan and extract features
        scan_result = scanner.scan_directory(source_path)
        files_with_features = extractor.extract_batch(scan_result.files[:max_files])
        
        click.echo(f"📁 Analyzing {len(files_with_features)} files (limited for cost control)")
        
        # Initialize LLM engine
        config = LLMConfig(provider=provider, model=model, cost_limit_usd=2.0)  # Low limit for preview
        engine = LLMOrganizationEngine(config)
        
        # Analyze files
        click.echo("🔍 Performing LLM analysis...")
        file_summaries = engine._analyze_file_contents(files_with_features)
        
        # Display results
        click.echo(f"\n🤖 LLM Analysis Results:")
        
        categories = {}
        for summary in file_summaries:
            category = summary.get('category', 'Unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(summary)
        
        click.echo(f"\n📊 Detected Categories:")
        for category, files in categories.items():
            click.echo(f"  📂 {category}: {len(files)} files")
        
        if show_analysis:
            click.echo(f"\n📄 Detailed Analysis:")
            for category, files in categories.items():
                click.echo(f"\n  📂 {category}:")
                for file_summary in files[:3]:
                    click.echo(f"    📄 {file_summary['file_name']}")
                    click.echo(f"       Summary: {file_summary.get('summary', 'N/A')[:80]}...")
                    click.echo(f"       Topics: {', '.join(file_summary.get('topics', []))}")
                    click.echo(f"       Confidence: {file_summary.get('confidence', 0):.2f}")
                
                if len(files) > 3:
                    click.echo(f"    ... and {len(files) - 3} more files")
        
        # Generate folder structure preview
        click.echo(f"\n🏗️  Generating folder structure...")
        folder_structure = engine._generate_folder_structure(file_summaries)
        
        click.echo(f"\n📁 Proposed Folder Structure:")
        for directory in folder_structure.get('directories', []):
            click.echo(f"  📂 {directory}")
        
        if folder_structure.get('reasoning'):
            click.echo(f"\n💭 LLM Reasoning: {folder_structure['reasoning']}")
        
        # Show cost
        total_cost = engine.cost_tracker.total_cost_usd
        click.echo(f"\n💰 Preview Cost: ${total_cost:.4f}")
        
    except ImportError:
        click.echo("❌ LLM preview requires: pip install openai (or anthropic)")
    except Exception as e:
        click.echo(f"❌ LLM preview failed: {e}")

@cli.command()
def cost_report():
    """Show detailed LLM usage and cost report"""
    
    cost_tracker = CostTracker()
    report = cost_tracker.generate_report()
    
    click.echo("💰 LLM Cost Report")
    click.echo("=" * 50)
    click.echo(report)
    
    # Show daily breakdown
    daily_costs = cost_tracker.get_daily_costs(30)
    if daily_costs:
        click.echo("\n📅 Daily Usage (Last 30 Days):")
        for date, cost in sorted(daily_costs.items())[-10:]:  # Last 10 days
            click.echo(f"  {date}: ${cost:.4f}")

@cli.command()
@click.argument('source_path', type=click.Path(exists=True))
@click.option('--threshold', type=int, default=5, help='Similarity threshold for images')
def find_similar_images(source_path, threshold):
    """Find visually similar images using perceptual hashing"""
    
    click.echo("🖼️  Finding similar images...")
    
    try:
        from ..engines.advanced_features import ImageHashExtractor
        
        # Setup
        scanner = FileScanner()
        scan_result = scanner.scan_directory(source_path)
        
        # Extract hashes
        extractor = ImageHashExtractor()
        similar_groups = extractor.find_similar_images(scan_result.files, threshold)
        
        if not similar_groups:
            click.echo("✅ No similar images found!")
            return
        
        click.echo(f"\n🔍 Found {len(similar_groups)} groups of similar images:")
        
        total_duplicates = 0
        total_wasted_space = 0
        
        for i, group in enumerate(similar_groups, 1):
            click.echo(f"\n📸 Group {i}: {len(group)} similar images")
            
            for file_info in group:
                click.echo(f"  📄 {file_info.name} ({file_info.size_mb:.2f} MB)")
            
            # Calculate potential space savings
            if len(group) > 1:
                avg_size = sum(f.size for f in group) / len(group)
                wasted = avg_size * (len(group) - 1)
                total_wasted_space += wasted
                total_duplicates += len(group) - 1
                
                click.echo(f"  💾 Potential savings: {wasted / (1024*1024):.2f} MB")
        
        click.echo(f"\n📊 Summary:")
        click.echo(f"  🔄 Total similar images: {total_duplicates}")
        click.echo(f"  💾 Potential space savings: {total_wasted_space / (1024*1024):.2f} MB")
        
    except ImportError:
        click.echo("❌ Image similarity requires: pip install imagehash Pillow")
    except Exception as e:
        click.echo(f"❌ Image analysis failed: {e}")

@cli.command()
@click.option('--config-file', help='Path to configuration file')
def setup():
    """Setup File Butler configuration"""
    
    from ..utils.config import Config
    
    click.echo("⚙️  Setting up AI File Butler configuration...")
    
    # Create sample config
    config = Config(config_file=config_file)
    config.create_sample_config()
    
    click.echo("\n🔧 Configuration options:")
    click.echo("  1. Edit the configuration file to customize behavior")
    click.echo("  2. Set environment variables for API keys:")
    click.echo("     export OPENAI_API_KEY=your_openai_key")
    click.echo("     export ANTHROPIC_API_KEY=your_anthropic_key")
    click.echo("  3. Install optional dependencies:")
    click.echo("     pip install scikit-learn pandas numpy  # For ML features")
    click.echo("     pip install openai anthropic          # For LLM features")
    click.echo("     pip install imagehash                 # For image similarity")

@cli.command()
def benchmark():
    """Benchmark different organization strategies"""
    
    click.echo("⚡ Benchmarking organization strategies...")
    
    # Create test data
    from demo.create_test_data import TestDataGenerator
    
    generator = TestDataGenerator()
    test_path = generator.create_messy_downloads_scenario()
    
    click.echo(f"📁 Created test data: {test_path}")
    
    strategies = ['rules', 'clustering', 'llm']
    results = {}
    
    for strategy in strategies:
        if strategy == 'llm' and not os.getenv('OPENAI_API_KEY'):
            click.echo(f"⏭️  Skipping {strategy} (no API key)")
            continue
        
        click.echo(f"\n🧪 Testing {strategy} strategy...")
        
        try:
            import time
            start_time = time.time()
            
            butler = EnhancedFileButlerCLI()
            plan = butler.organize_directory(
                source_path=test_path,
                strategy=strategy,
                dry_run=True,
                verbose=False
            )
            
            execution_time = time.time() - start_time
            
            results[strategy] = {
                'time': execution_time,
                'actions': len(plan.actions),
                'directories': len([a for a in plan.actions if a.action_type.value == 'create_dir'])
            }
            
            click.echo(f"  ⏱️  Time: {execution_time:.2f}s")
            click.echo(f"  📊 Actions: {results[strategy]['actions']}")
            click.echo(f"  📁 Directories: {results[strategy]['directories']}")
            
        except Exception as e:
            click.echo(f"  ❌ Failed: {e}")
    
    # Summary
    if results:
        click.echo(f"\n📊 Benchmark Summary:")
        click.echo(f"{'Strategy':<12} {'Time (s)':<10} {'Actions':<10} {'Dirs':<10}")
        click.echo("-" * 45)
        
        for strategy, data in results.items():
            click.echo(f"{strategy:<12} {data['time']:<10.2f} {data['actions']:<10} {data['directories']:<10}")

if __name__ == '__main__':
    cli()