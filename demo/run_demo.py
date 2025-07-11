#!/usr/bin/env python3
# demo/run_demo.py

"""
AI File Butler Demo Script

This script demonstrates the capabilities of the AI File Butler by:
1. Creating realistic test data 
2. Running the organization process
3. Showing before/after comparisons
4. Demonstrating different organization strategies
"""

import os
import sys
from pathlib import Path
import time

# Add parent directory to path so we can import file_butler
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.create_test_data import TestDataGenerator
from file_butler.core.scanner import FileScanner
from file_butler.core.extractor import FeatureExtractorManager
from file_butler.core.executor import ActionExecutor
from file_butler.engines.rules import RuleBasedEngine
from file_butler.detectors.duplicates import DuplicateDetector, NearDuplicateDetector

class FileButlerDemo:
    """Demonstrates File Butler capabilities"""
    
    def __init__(self):
        self.generator = TestDataGenerator()
        self.scanner = FileScanner()
        self.extractor = FeatureExtractorManager()
        self.rules_engine = RuleBasedEngine()
        self.duplicate_detector = DuplicateDetector()
        self.near_duplicate_detector = NearDuplicateDetector()
    
    def run_full_demo(self):
        """Run complete demonstration"""
        print("🤖 AI File Butler - Full Demonstration")
        print("=" * 60)
        
        # Create test scenarios
        scenarios = self._create_test_scenarios()
        
        for scenario_name, scenario_path in scenarios.items():
            print(f"\n🎭 Demo Scenario: {scenario_name}")
            print("-" * 40)
            
            # Run organization demo
            self._demo_organization(scenario_path)
            
            # Wait for user input before next scenario
            if len(scenarios) > 1:
                input("\nPress Enter to continue to next scenario...")
        
        print("\n🎉 Demo complete!")
        print("💡 Tip: Run 'python -m file_butler.interfaces.cli demo' for interactive CLI demo")
    
    def _create_test_scenarios(self):
        """Create different test scenarios"""
        print("📁 Creating test data scenarios...")
        
        scenarios = {}
        
        try:
            # Messy downloads scenario
            path = self.generator.create_messy_downloads_scenario()
            scenarios["Messy Downloads Folder"] = path
            print(f"✅ Created: Messy Downloads ({path})")
            
            # Photo collection scenario  
            path = self.generator.create_photo_collection_scenario()
            scenarios["Photo Collection"] = path
            print(f"✅ Created: Photo Collection ({path})")
            
            # Document archive scenario
            path = self.generator.create_document_archive_scenario()
            scenarios["Document Archive"] = path
            print(f"✅ Created: Document Archive ({path})")
            
        except Exception as e:
            print(f"❌ Error creating test data: {e}")
            return {"Messy Downloads Folder": self.generator.create_messy_downloads_scenario()}
        
        return scenarios
    
    def _demo_organization(self, scenario_path: str):
        """Demonstrate organization for a single scenario"""
        
        # Step 1: Show initial state
        print(f"📊 Analyzing initial state...")
        self._show_directory_analysis(scenario_path, "BEFORE")
        
        # Step 2: Run file organization
        print(f"\n🧠 Running AI organization...")
        plan = self._organize_files(scenario_path)
        
        # Step 3: Show organization plan
        self._show_organization_plan(plan)
        
        # Step 4: Execute (dry run)
        print(f"\n🔄 Executing organization plan (DRY RUN)...")
        self._execute_plan(plan, dry_run=True)
        
        # Step 5: Ask if user wants to see actual execution
        response = input("\n❓ Execute the plan for real? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            print(f"\n🚀 Executing organization plan...")
            result = self._execute_plan(plan, dry_run=False)
            
            # Show results
            organized_path = Path(scenario_path) / "organized"
            if organized_path.exists():
                print(f"\n📊 Analyzing organized result...")
                self._show_directory_analysis(str(organized_path), "AFTER")
        else:
            print("⏭️  Skipping actual execution")
    
    def _show_directory_analysis(self, directory_path: str, label: str):
        """Show analysis of directory contents"""
        
        print(f"\n📋 Directory Analysis - {label}")
        print(f"📁 Path: {directory_path}")
        
        try:
            # Scan directory
            scan_result = self.scanner.scan_directory(directory_path)
            
            if not scan_result.files:
                print("📄 No files found")
                return
            
            # Basic statistics
            print(f"📊 Files: {len(scan_result.files)}")
            print(f"💾 Total size: {scan_result.total_size_mb:.2f} MB")
            
            # File type distribution
            print(f"\n📈 File types:")
            for file_type, count in scan_result.file_types_distribution.items():
                print(f"  {file_type.value.title()}: {count}")
            
            # Find duplicates
            duplicates = self.duplicate_detector.find_exact_duplicates(scan_result.files)
            if duplicates:
                stats = self.duplicate_detector.get_duplicate_statistics(duplicates)
                print(f"\n🔄 Duplicates found:")
                print(f"  Groups: {stats['duplicate_groups']}")
                print(f"  Duplicate files: {stats['total_duplicates']}")
                print(f"  Wasted space: {stats['wasted_space_mb']:.2f} MB")
            else:
                print(f"\n✅ No duplicates found")
            
            # Show sample files
            print(f"\n📄 Sample files:")
            for i, file_info in enumerate(scan_result.files[:5]):
                print(f"  {i+1}. {file_info.name} ({file_info.size_mb:.2f} MB)")
            
            if len(scan_result.files) > 5:
                print(f"  ... and {len(scan_result.files) - 5} more files")
                
        except Exception as e:
            print(f"❌ Error analyzing directory: {e}")
    
    def _organize_files(self, directory_path: str):
        """Run the organization process"""
        
        try:
            # Scan files
            scan_result = self.scanner.scan_directory(directory_path)
            print(f"  📁 Scanned {len(scan_result.files)} files")
            
            # Extract features
            files_with_features = self.extractor.extract_batch(scan_result.files)
            print(f"  🔬 Extracted features for {len(files_with_features)} files")
            
            # Find duplicates
            duplicates = self.duplicate_detector.find_exact_duplicates(files_with_features)
            if duplicates:
                print(f"  🔄 Found {len(duplicates)} groups of duplicates")
            
            # Generate organization plan
            plan = self.rules_engine.organize(files_with_features)
            plan.source_directory = directory_path
            
            # Add duplicate deletion suggestions
            if duplicates:
                deletion_actions = self.duplicate_detector.suggest_deletions(duplicates)
                plan.actions.extend(deletion_actions)
                print(f"  🗑️  Added {len(deletion_actions)} deletion suggestions")
            
            print(f"  📋 Generated {len(plan.actions)} total actions")
            
            return plan
            
        except Exception as e:
            print(f"❌ Error during organization: {e}")
            raise
    
    def _show_organization_plan(self, plan):
        """Display the organization plan"""
        
        summary = plan.summary()
        
        print(f"\n📋 Organization Plan Summary:")
        print(f"  📁 Directories to create: {summary['directories_to_create']}")
        print(f"  📦 Files to move: {summary['files_to_move']}")
        print(f"  🗑️  Files to delete: {summary['files_to_delete']}")
        print(f"  📊 Total actions: {summary['total_actions']}")
        
        # Show sample actions by type
        create_actions = [a for a in plan.actions if a.action_type.value == 'create_dir']
        move_actions = [a for a in plan.actions if a.action_type.value == 'move']
        delete_actions = [a for a in plan.actions if a.action_type.value == 'delete']
        
        if create_actions:
            print(f"\n📁 Sample directories to create:")
            for action in create_actions[:3]:
                dir_name = Path(action.destination).name
                print(f"  📂 {dir_name}")
            if len(create_actions) > 3:
                print(f"  ... and {len(create_actions) - 3} more directories")
        
        if move_actions:
            print(f"\n📦 Sample file movements:")
            for action in move_actions[:5]:
                source_name = Path(action.source).name
                target_dir = Path(action.destination).parent.name
                print(f"  📄 {source_name} → {target_dir}/")
            if len(move_actions) > 5:
                print(f"  ... and {len(move_actions) - 5} more moves")
        
        if delete_actions:
            print(f"\n🗑️  Sample deletion suggestions:")
            for action in delete_actions[:3]:
                file_name = Path(action.source).name
                print(f"  ❌ {file_name} ({action.reason})")
            if len(delete_actions) > 3:
                print(f"  ... and {len(delete_actions) - 3} more deletions")
    
    def _execute_plan(self, plan, dry_run=True):
        """Execute the organization plan"""
        
        executor = ActionExecutor(dry_run=dry_run)
        
        def progress_callback(current, total, action_desc):
            if current % 5 == 0 or current == total:  # Update every 5 actions
                progress = (current / total) * 100
                print(f"\r  ⏳ Progress: {progress:.1f}% ({current}/{total})", end='', flush=True)
        
        result = executor.execute_plan(plan)
        
        print()  # New line after progress
        print(f"  ✅ Execution complete!")
        print(f"  ✓ Actions executed: {result.executed_actions}")
        if result.failed_actions > 0:
            print(f"  ❌ Actions failed: {result.failed_actions}")
        print(f"  📈 Success rate: {result.success_rate:.1f}%")
        print(f"  ⏱️  Time: {result.execution_time:.2f} seconds")
        
        return result
    
    def demo_specific_features(self):
        """Demonstrate specific features in isolation"""
        
        print("🔍 Feature-Specific Demonstrations")
        print("=" * 50)
        
        # Create a small test dataset
        test_path = self.generator.create_messy_downloads_scenario()
        
        # Demo 1: File scanning and feature extraction
        print("\n1️⃣  File Scanning and Feature Extraction")
        print("-" * 30)
        
        scan_result = self.scanner.scan_directory(test_path)
        print(f"📁 Scanned {len(scan_result.files)} files in {scan_result.scan_time:.2f}s")
        
        # Show feature extraction for one file
        if scan_result.files:
            sample_file = scan_result.files[0]
            features = self.extractor.extract_features(sample_file)
            
            print(f"\n📄 Sample file: {sample_file.name}")
            print(f"📊 Extracted features:")
            for extractor_name, feature_dict in features.items():
                print(f"  {extractor_name}: {len(feature_dict)} features")
                # Show a few sample features
                for key, value in list(feature_dict.items())[:3]:
                    print(f"    {key}: {value}")
        
        # Demo 2: Duplicate detection
        print("\n2️⃣  Duplicate Detection")
        print("-" * 25)
        
        duplicates = self.duplicate_detector.find_exact_duplicates(scan_result.files)
        if duplicates:
            print(f"🔄 Found {len(duplicates)} groups of duplicates")
            for i, group in enumerate(duplicates[:2]):  # Show first 2 groups
                print(f"  Group {i+1}: {len(group)} identical files")
                for file_info in group:
                    print(f"    📄 {file_info.name}")
        else:
            print("✅ No exact duplicates found")
        
        # Demo 3: Rule-based organization
        print("\n3️⃣  Rule-Based Organization")
        print("-" * 30)
        
        # Extract features first
        files_with_features = self.extractor.extract_batch(scan_result.files[:5])  # Just first 5 files
        
        print("📋 Applying organization rules to sample files:")
        for file_info in files_with_features:
            # Apply rules manually to show decision process
            target_path = self.rules_engine._apply_rules(file_info)
            print(f"  📄 {file_info.name}")
            print(f"    Type: {file_info.file_type.value}")
            print(f"    Target: {target_path}")
        
        print(f"\n✅ Feature demonstrations complete!")


def main():
    """Main demo function"""
    
    demo = FileButlerDemo()
    
    print("Welcome to the AI File Butler Demo!")
    print("What would you like to see?")
    print()
    print("1. Full Demo (create test data + organize)")
    print("2. Feature-Specific Demos")
    print("3. Quick Test (messy downloads only)")
    print()
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            demo.run_full_demo()
            break
        elif choice == '2':
            demo.demo_specific_features()
            break
        elif choice == '3':
            # Quick demo with just messy downloads
            test_path = demo.generator.create_messy_downloads_scenario()
            demo._demo_organization(test_path)
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()