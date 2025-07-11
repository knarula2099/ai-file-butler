#!/usr/bin/env python3
# demo/test_custom_prompts.py

"""
Test script for custom prompt functionality
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from file_butler.core.scanner import FileScanner
from file_butler.core.extractor import FeatureExtractorManager
from file_butler.engines.llm import LLMOrganizationEngine, LLMConfig

def test_custom_prompts():
    """Test different custom prompts"""
    
    # Check if LLM is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("💡 Set it with: export OPENAI_API_KEY=your_key_here")
        return
    
    # Setup
    scanner = FileScanner()
    extractor = FeatureExtractorManager()
    
    # Create a simple test directory with some files
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Create some test files with different names that suggest organization
    test_files = [
        "vacation_paris_2024.jpg",
        "vacation_tokyo_2023.jpg", 
        "vacation_newyork_2024.jpg",
        "work_project_alpha_report.pdf",
        "work_project_beta_notes.txt",
        "personal_recipe_pasta.txt",
        "personal_recipe_salad.txt"
    ]
    
    for filename in test_files:
        file_path = test_dir / filename
        file_path.write_text(f"Test content for {filename}")
    
    print(f"📁 Created test files in {test_dir}")
    
    # Scan files
    scan_result = scanner.scan_directory(str(test_dir))
    files_with_features = extractor.extract_batch(scan_result.files)
    
    print(f"🔍 Found {len(files_with_features)} files")
    
    # Test different custom prompts
    prompts = [
        {
            "name": "Organize by Year",
            "prompt": "Organize these files by the year they were created or modified. Create folders like '2024', '2023', '2022', etc. Look for dates in file names, metadata, or content.",
            "expected": "Should create year-based folders"
        },
        {
            "name": "Organize by Location",
            "prompt": "Organize these files by geographic location. Extract location information from file names, metadata, or content. Create folders like 'Paris', 'Tokyo', 'New York', etc.",
            "expected": "Should create location-based folders"
        },
        {
            "name": "Organize by Category",
            "prompt": "Organize these files by their main category. Create folders like 'Vacation', 'Work', 'Personal' based on the content and file names.",
            "expected": "Should create category-based folders"
        }
    ]
    
    for i, prompt_info in enumerate(prompts, 1):
        print(f"\n🎯 Test {i}: {prompt_info['name']}")
        print(f"   📝 Expected: {prompt_info['expected']}")
        
        try:
            # Initialize LLM engine with custom prompt
            config = LLMConfig(
                provider='openai',
                model='gpt-3.5-turbo',
                cost_limit_usd=2.0,
                custom_prompt=prompt_info['prompt']
            )
            engine = LLMOrganizationEngine(config)
            
            # Generate organization plan
            plan = engine.organize(files_with_features)
            
            # Show results
            print(f"   📋 Generated {len(plan.actions)} actions")
            
            # Show directory creation actions
            dir_actions = [a for a in plan.actions if a.action_type.value == 'create_dir']
            if dir_actions:
                print(f"   📁 Directories to create:")
                for action in dir_actions:
                    print(f"      📂 {action.destination}")
            
            # Show move actions
            move_actions = [a for a in plan.actions if a.action_type.value == 'move']
            if move_actions:
                print(f"   📄 Files to move:")
                for action in move_actions[:5]:  # Show first 5
                    source_name = Path(action.source).name
                    target_dir = Path(action.destination).parent.name
                    print(f"      📄 {source_name} → {target_dir}")
                
                if len(move_actions) > 5:
                    print(f"      ... and {len(move_actions) - 5} more")
            
            # Show cost
            total_cost = engine.cost_tracker.total_cost_usd
            print(f"   💰 Cost: ${total_cost:.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print(f"\n🧹 Cleaned up test directory")
    
    print(f"\n💡 Try the CLI command:")
    print(f"   file-butler organize-with-prompt /path/to/files --prompt \"Organize by year\"")
    print(f"   file-butler prompt-examples")

if __name__ == "__main__":
    test_custom_prompts() 