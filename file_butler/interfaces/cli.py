# file_butler/interfaces/cli.py
import click
import time
from pathlib import Path
from typing import Optional

from ..core.scanner import FileScanner
from ..core.extractor import FeatureExtractorManager
from ..core.executor import ActionExecutor
from ..engines.rules import RuleBasedEngine
from ..detectors.duplicates import DuplicateDetector

class FileButlerCLI:
    """Command Line Interface for File Butler"""
    
    def __init__(self):
        self.scanner = FileScanner()
        self.extractor = FeatureExtractorManager()
        self.engine = RuleBasedEngine()
        self.executor = ActionExecutor()
    
    def organize_directory(self, 
                          source_path: str,
                          target_path: Optional[str] = None,
                          dry_run: bool = True,
                          find_duplicates: bool = True,
                          verbose: bool = False):
        """Main organization workflow"""
        
        # Setup
        source_path = Path(source_path).resolve()
        if target_path is None:
            target_path = source_path / "organized"
        else:
            target_path = Path(target_path).resolve()
        
        if verbose:
            click.echo(f"🤖 AI File Butler")
            click.echo(f"📁 Source: {source_path}")
            click.echo(f"📂 Target: {target_path}")
            click.echo(f"🔄 Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
            click.echo("-" * 60)
        
        # Step 1: Scan directory
        if verbose:
            click.echo("🔍 Scanning directory...")
        
        def progress_callback(count, current_file):
            if verbose and count % 10 == 0:
                click.echo(f"  Found {count} files...", nl=False)
                click.echo(f"\r", nl=False)
        
        scan_result = self.scanner.scan_directory(str(source_path))
        
        if verbose:
            click.echo(f"✅ Scan complete: {len(scan_result.files)} files found")
            if scan_result.errors:
                click.echo(f"⚠️  {len(scan_result.errors)} errors encountered")
        
        if not scan_result.files:
            click.echo("❌ No files found to organize!")
            return
        
        # Step 2: Extract features
        if verbose:
            click.echo("🔬 Extracting features...")
        
        files_with_features = self.extractor.extract_batch(scan_result.files)
        
        # Step 3: Find duplicates (optional)
        duplicate_groups = []
        if find_duplicates:
            if verbose:
                click.echo("🔍 Finding duplicates...")
            
            detector = DuplicateDetector()
            duplicate_groups = detector.find_exact_duplicates(files_with_features)
            
            if verbose and duplicate_groups:
                click.echo(f"🔄 Found {len(duplicate_groups)} groups of duplicates")
        
        # Step 4: Generate organization plan
        if verbose:
            click.echo("🧠 Generating organization plan...")
        
        plan = self.engine.organize(files_with_features)
        plan.source_directory = str(source_path)
        plan.target_directory = str(target_path)
        
        # Add duplicate deletion suggestions
        if duplicate_groups:
            detector = DuplicateDetector()
            deletion_actions = detector.suggest_deletions(duplicate_groups)
            plan.actions.extend(deletion_actions)
        
        # Step 5: Display plan summary
        self._display_plan_summary(plan, verbose)
        
        # Step 6: Execute plan
        if not dry_run:
            if click.confirm("Proceed with execution?"):
                self._execute_plan(plan, verbose)
            else:
                click.echo("❌ Execution cancelled")
        else:
            click.echo("\n💡 Run with --execute to apply changes")
        
        return plan
    
    def _display_plan_summary(self, plan, verbose=True):
        """Display organization plan summary"""
        
        summary = plan.summary()
        
        if verbose:
            click.echo("\n📋 Organization Plan Summary:")
            click.echo(f"  📁 Directories to create: {summary['directories_to_create']}")
            click.echo(f"  📦 Files to move: {summary['files_to_move']}")
            click.echo(f"  🗑️  Files to delete: {summary['files_to_delete']}")
            click.echo(f"  📊 Total actions: {summary['total_actions']}")
        
        if summary['files_to_delete'] > 0:
            click.echo(f"\n⚠️  {summary['files_to_delete']} files marked for deletion (duplicates)")
            
        # Show sample actions
        if verbose and plan.actions:
            click.echo("\n📄 Sample actions:")
            for i, action in enumerate(plan.actions[:5]):
                click.echo(f"  {i+1}. {action}")
            
            if len(plan.actions) > 5:
                click.echo(f"  ... and {len(plan.actions) - 5} more actions")
    
    def _execute_plan(self, plan, verbose=True):
        """Execute the organization plan"""
        
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
            click.echo(f"  ❌ Actions failed: {result.failed_actions}")
            click.echo(f"  📈 Success rate: {result.success_rate:.1f}%")
            click.echo(f"  ⏱️  Time: {result.execution_time:.2f} seconds")
        
        return result

# CLI Commands
@click.group()
def cli():
    """AI File Butler - Intelligent file organization tool"""
    pass

@cli.command()
@click.argument('source_path', type=click.Path(exists=True))
@click.option('--target', '-t', type=click.Path(), help='Target directory for organized files')
@click.option('--execute', is_flag=True, help='Execute the plan (default is dry run)')
@click.option('--no-duplicates', is_flag=True, help='Skip duplicate detection')
@click.option('--quiet', '-q', is_flag=True, help='Minimal output')
def organize(source_path, target, execute, no_duplicates, quiet):
    """Organize files in the specified directory"""
    
    butler = FileButlerCLI()
    
    try:
        plan = butler.organize_directory(
            source_path=source_path,
            target_path=target,
            dry_run=not execute,
            find_duplicates=not no_duplicates,
            verbose=not quiet
        )
        
        if not quiet:
            click.echo("\n🎉 File organization complete!")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        raise click.Abort()

@cli.command()
@click.argument('source_path', type=click.Path(exists=True))
@click.option('--detailed', is_flag=True, help='Show detailed file information')
def scan(source_path, detailed):
    """Scan directory and show file statistics"""
    
    scanner = FileScanner()
    result = scanner.scan_directory(source_path)
    
    click.echo(f"📁 Directory: {source_path}")
    click.echo(f"📊 Files found: {len(result.files)}")
    click.echo(f"💾 Total size: {result.total_size_mb:.2f} MB")
    click.echo(f"⏱️  Scan time: {result.scan_time:.2f} seconds")
    
    if result.errors:
        click.echo(f"⚠️  Errors: {len(result.errors)}")
    
    # File type distribution
    click.echo("\n📈 File type distribution:")
    for file_type, count in result.file_types_distribution.items():
        click.echo(f"  {file_type.value}: {count}")
    
    if detailed:
        click.echo("\n📄 File details:")
        for file_info in result.files[:20]:  # Show first 20
            click.echo(f"  {file_info.name} ({file_info.size_mb:.2f} MB)")
        
        if len(result.files) > 20:
            click.echo(f"  ... and {len(result.files) - 20} more files")

@cli.command()
@click.argument('source_path', type=click.Path(exists=True))
def duplicates(source_path):
    """Find duplicate files in directory"""
    
    scanner = FileScanner()
    detector = DuplicateDetector()
    
    click.echo("🔍 Scanning for duplicates...")
    scan_result = scanner.scan_directory(source_path)
    
    click.echo("🔬 Analyzing files...")
    duplicate_groups = detector.find_exact_duplicates(scan_result.files)
    
    if not duplicate_groups:
        click.echo("✅ No exact duplicates found!")
        return
    
    click.echo(f"\n🔄 Found {len(duplicate_groups)} groups of duplicates:")
    
    total_duplicates = 0
    total_wasted_space = 0
    
    for i, group in enumerate(duplicate_groups, 1):
        click.echo(f"\nGroup {i}: {len(group)} files")
        for file_info in group:
            click.echo(f"  📄 {file_info.name} ({file_info.size_mb:.2f} MB)")
        
        # Calculate wasted space (all but one file)
        wasted_files = len(group) - 1
        wasted_space = group[0].size_mb * wasted_files
        
        total_duplicates += wasted_files
        total_wasted_space += wasted_space
        
        click.echo(f"  💾 Wasted space: {wasted_space:.2f} MB")
    
    click.echo(f"\n📊 Summary:")
    click.echo(f"  🔄 Total duplicate files: {total_duplicates}")
    click.echo(f"  💾 Total wasted space: {total_wasted_space:.2f} MB")

@cli.command()
def demo():
    """Run demo with generated test data"""
    from demo.create_test_data import TestDataGenerator
    
    generator = TestDataGenerator()
    
    click.echo("🎭 Creating demo data...")
    scenario_path = generator.create_messy_downloads_scenario()
    
    click.echo(f"📁 Demo data created at: {scenario_path}")
    click.echo("\n🤖 Running organization demo...")
    
    butler = FileButlerCLI()
    plan = butler.organize_directory(
        source_path=scenario_path,
        dry_run=True,
        find_duplicates=True,
        verbose=True
    )
    
    click.echo("\n🎉 Demo complete! Check the generated plan above.")
    click.echo(f"💡 To execute: file-butler organize {scenario_path} --execute")

if __name__ == '__main__':
    cli()