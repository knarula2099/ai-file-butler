# file_butler/engines/llm.py
import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

from ..core.models import FileInfo, Action, ActionType, OrganizationPlan, FileType
from ..utils.caching import LLMCache
from ..utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM engine"""
    provider: str = "openai"  # "openai", "anthropic", "local"
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.2
    max_file_size_kb: int = 50  # Max file size to send to LLM
    batch_size: int = 5  # Files to analyze together
    enable_caching: bool = True
    cost_limit_usd: float = 10.0  # Safety limit

class LLMOrganizationEngine:
    """LLM-powered intelligent file organization"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.cache = LLMCache() if self.config.enable_caching else None
        self.cost_tracker = CostTracker()
        self.client = self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.config.provider == "openai":
            try:
                import openai
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                return openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        elif self.config.provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                return anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        elif self.config.provider == "local":
            # For local models using Ollama or similar
            return self._initialize_local_client()
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    def _initialize_local_client(self):
        """Initialize local LLM client (Ollama, etc.)"""
        try:
            import requests
            # Test if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                return {"type": "ollama", "base_url": "http://localhost:11434"}
            else:
                logger.warning("Local LLM server not accessible")
                return None
        except Exception as e:
            logger.warning(f"Cannot connect to local LLM: {e}")
            return None
    
    def organize(self, files: List[FileInfo]) -> OrganizationPlan:
        """Generate organization plan using LLM"""
        
        logger.info(f"Starting LLM organization for {len(files)} files")
        
        # Check cost limits
        if self.cost_tracker.total_cost_usd >= self.config.cost_limit_usd:
            logger.warning(f"Cost limit reached: ${self.cost_tracker.total_cost_usd:.2f}")
            raise Exception(f"Cost limit of ${self.config.cost_limit_usd} reached")
        
        actions = []
        
        # Step 1: Analyze file contents and extract summaries
        file_summaries = self._analyze_file_contents(files)
        
        # Step 2: Generate folder structure based on content
        folder_structure = self._generate_folder_structure(file_summaries)
        
        # Step 3: Create directory actions
        for folder_path in folder_structure.get('directories', []):
            actions.append(Action(
                action_type=ActionType.CREATE_DIR,
                source="",
                destination=folder_path,
                reason="LLM-suggested directory structure"
            ))
        
        # Step 4: Generate file mappings
        file_mappings = self._map_files_to_folders(file_summaries, folder_structure)
        
        # Step 5: Create move actions
        for mapping in file_mappings:
            actions.append(Action(
                action_type=ActionType.MOVE,
                source=mapping['source_path'],
                destination=mapping['target_path'],
                reason=mapping['reason'],
                confidence=mapping['confidence']
            ))
        
        logger.info(f"Generated {len(actions)} actions with LLM engine")
        logger.info(f"Total cost so far: ${self.cost_tracker.total_cost_usd:.4f}")
        
        return OrganizationPlan(
            actions=actions,
            source_directory="",  # Will be set by caller
            target_directory="organized"
        )
    
    def _analyze_file_contents(self, files: List[FileInfo]) -> List[Dict[str, Any]]:
        """Analyze file contents and generate summaries"""
        
        summaries = []
        
        # Group files by type for batch processing
        text_files = [f for f in files if f.file_type in [FileType.DOCUMENT, FileType.CODE]]
        other_files = [f for f in files if f not in text_files]
        
        # Process text files with content analysis
        if text_files:
            text_summaries = self._analyze_text_files_batch(text_files)
            summaries.extend(text_summaries)
        
        # Process other files with metadata analysis
        if other_files:
            other_summaries = self._analyze_other_files(other_files)
            summaries.extend(other_summaries)
        
        return summaries
    
    def _analyze_text_files_batch(self, files: List[FileInfo]) -> List[Dict[str, Any]]:
        """Analyze text files in batches for efficiency"""
        
        summaries = []
        
        # Process files in batches
        for i in range(0, len(files), self.config.batch_size):
            batch = files[i:i + self.config.batch_size]
            batch_summaries = self._process_text_batch(batch)
            summaries.extend(batch_summaries)
        
        return summaries
    
    def _process_text_batch(self, files: List[FileInfo]) -> List[Dict[str, Any]]:
        """Process a batch of text files"""
        
        batch_data = []
        
        for file_info in files:
            # Check cache first
            cache_key = self._get_cache_key(file_info)
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for {file_info.name}")
                    batch_data.append(cached_result)
                    continue
            
            # Read file content
            content = self._read_file_content(file_info)
            if not content:
                # Fallback to metadata-based analysis
                summary = self._analyze_file_metadata(file_info)
            else:
                # LLM analysis
                summary = self._analyze_content_with_llm(file_info, content)
            
            # Cache the result
            if self.cache:
                self.cache.set(cache_key, summary)
            
            batch_data.append(summary)
        
        return batch_data
    
    def _analyze_content_with_llm(self, file_info: FileInfo, content: str) -> Dict[str, Any]:
        """Analyze file content using LLM"""
        
        prompt = self._create_analysis_prompt(file_info.name, content)
        
        try:
            response = self._call_llm(prompt, max_tokens=200)
            
            # Parse structured response
            analysis = self._parse_analysis_response(response)
            
            return {
                'file_name': file_info.name,
                'file_path': file_info.path,
                'file_type': file_info.file_type.value,
                'summary': analysis.get('summary', ''),
                'category': analysis.get('category', 'General'),
                'topics': analysis.get('topics', []),
                'importance': analysis.get('importance', 'medium'),
                'suggested_folder': analysis.get('suggested_folder', 'Documents'),
                'confidence': analysis.get('confidence', 0.7)
            }
            
        except Exception as e:
            logger.warning(f"LLM analysis failed for {file_info.name}: {e}")
            return self._analyze_file_metadata(file_info)
    
    def _create_analysis_prompt(self, filename: str, content: str) -> str:
        """Create prompt for file content analysis"""
        
        # Truncate content if too long
        max_chars = 2000
        if len(content) > max_chars:
            content = content[:max_chars] + "... [truncated]"
        
        prompt = f"""Analyze this file and provide a structured response in JSON format.

File: {filename}
Content: {content}

Please respond with a JSON object containing:
- "summary": Brief 1-2 sentence description of the file's content and purpose
- "category": Main category (Work, Personal, Financial, Educational, Creative, Technical, etc.)
- "topics": List of 2-3 key topics or keywords
- "importance": "high", "medium", or "low" based on apparent importance
- "suggested_folder": Suggested folder name for organization
- "confidence": Float between 0-1 indicating confidence in categorization

Example response:
{{
    "summary": "Meeting notes about Q3 budget planning with action items for the finance team.",
    "category": "Work",
    "topics": ["budget", "planning", "meetings"],
    "importance": "high",
    "suggested_folder": "Work/Meetings",
    "confidence": 0.9
}}

Respond with only the JSON object:"""

        return prompt
    
    def _call_llm(self, prompt: str, max_tokens: int = None) -> str:
        """Call the configured LLM with the prompt"""
        
        max_tokens = max_tokens or self.config.max_tokens
        
        if self.config.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.config.temperature
            )
            
            # Track costs
            self.cost_tracker.add_openai_usage(
                model=self.config.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens
            )
            
            return response.choices[0].message.content.strip()
        
        elif self.config.provider == "anthropic":
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Track costs
            self.cost_tracker.add_anthropic_usage(
                model=self.config.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )
            
            return response.content[0].text.strip()
        
        elif self.config.provider == "local":
            return self._call_local_llm(prompt, max_tokens)
        
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _call_local_llm(self, prompt: str, max_tokens: int) -> str:
        """Call local LLM (Ollama)"""
        import requests
        
        if not self.client or self.client["type"] != "ollama":
            raise Exception("Local LLM not available")
        
        try:
            response = requests.post(
                f"{self.client['base_url']}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": self.config.temperature
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"Local LLM error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Local LLM call failed: {e}")
            raise
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        
        try:
            # Try to parse as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Fallback: parse manually
            logger.warning(f"Could not parse LLM response as JSON: {response[:100]}...")
            return {
                'summary': response[:100],
                'category': 'General',
                'topics': [],
                'importance': 'medium',
                'suggested_folder': 'Documents',
                'confidence': 0.5
            }
    
    def _generate_folder_structure(self, file_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate folder structure based on file analysis"""
        
        # Create prompt for folder structure generation
        categories = {}
        for summary in file_summaries:
            category = summary.get('category', 'General')
            if category not in categories:
                categories[category] = []
            categories[category].append({
                'name': summary['file_name'],
                'topics': summary.get('topics', []),
                'suggested_folder': summary.get('suggested_folder', 'Documents')
            })
        
        prompt = self._create_folder_structure_prompt(categories)
        
        try:
            response = self._call_llm(prompt, max_tokens=800)
            structure = json.loads(response)
            
            return {
                'directories': structure.get('directories', []),
                'reasoning': structure.get('reasoning', '')
            }
            
        except Exception as e:
            logger.warning(f"Folder structure generation failed: {e}")
            return self._create_fallback_structure(categories)
    
    def _create_folder_structure_prompt(self, categories: Dict[str, List]) -> str:
        """Create prompt for folder structure generation"""
        
        files_by_category = ""
        for category, files in categories.items():
            files_by_category += f"\n{category}:\n"
            for file_info in files[:5]:  # Limit to avoid token overflow
                topics_str = ", ".join(file_info['topics'][:3])
                files_by_category += f"  - {file_info['name']} (topics: {topics_str})\n"
            if len(files) > 5:
                files_by_category += f"  ... and {len(files) - 5} more files\n"
        
        prompt = f"""Based on these files and their analysis, create a logical folder structure for organization.

Files to organize:{files_by_category}

Create a hierarchical folder structure that:
1. Groups related files logically
2. Uses clear, descriptive folder names
3. Avoids too many nested levels (max 3 levels deep)
4. Follows common organization patterns

Respond with a JSON object containing:
- "directories": List of folder paths to create (e.g., ["Work/Projects", "Personal/Finance"])
- "reasoning": Brief explanation of the structure

Example:
{{
    "directories": [
        "Work/Projects",
        "Work/Meetings", 
        "Personal/Finance",
        "Personal/Health",
        "Documents/Reference"
    ],
    "reasoning": "Organized by primary context (Work/Personal) then by specific purpose."
}}

Respond with only the JSON object:"""

        return prompt
    
    def _map_files_to_folders(self, file_summaries: List[Dict[str, Any]], 
                             folder_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map files to appropriate folders"""
        
        mappings = []
        available_folders = folder_structure.get('directories', [])
        
        for summary in file_summaries:
            # Find best folder match
            best_folder = self._find_best_folder_match(summary, available_folders)
            
            file_name = summary['file_name']
            target_path = f"{best_folder}/{file_name}" if best_folder else file_name
            
            mappings.append({
                'source_path': summary['file_path'],
                'target_path': target_path,
                'reason': f"LLM categorized as {summary.get('category', 'General')}: {summary.get('summary', '')[:50]}...",
                'confidence': summary.get('confidence', 0.7)
            })
        
        return mappings
    
    def _find_best_folder_match(self, file_summary: Dict[str, Any], 
                               available_folders: List[str]) -> str:
        """Find the best folder match for a file"""
        
        file_category = file_summary.get('category', '').lower()
        file_topics = [t.lower() for t in file_summary.get('topics', [])]
        suggested_folder = file_summary.get('suggested_folder', '').lower()
        
        # Score each folder
        best_folder = ""
        best_score = 0
        
        for folder in available_folders:
            folder_lower = folder.lower()
            score = 0
            
            # Category match
            if file_category in folder_lower:
                score += 3
            
            # Topic matches
            for topic in file_topics:
                if topic in folder_lower:
                    score += 2
            
            # Suggested folder match
            if suggested_folder and suggested_folder in folder_lower:
                score += 4
            
            if score > best_score:
                best_score = score
                best_folder = folder
        
        # Fallback to default folders if no good match
        if not best_folder:
            if file_category in ['work', 'business', 'professional']:
                best_folder = next((f for f in available_folders if 'work' in f.lower()), 'Documents')
            elif file_category in ['personal', 'family', 'home']:
                best_folder = next((f for f in available_folders if 'personal' in f.lower()), 'Documents')
            else:
                best_folder = 'Documents'
        
        return best_folder
    
    def _analyze_other_files(self, files: List[FileInfo]) -> List[Dict[str, Any]]:
        """Analyze non-text files using metadata"""
        
        summaries = []
        for file_info in files:
            summary = self._analyze_file_metadata(file_info)
            summaries.append(summary)
        
        return summaries
    
    def _analyze_file_metadata(self, file_info: FileInfo) -> Dict[str, Any]:
        """Analyze file using metadata only (fallback)"""
        
        # Use rule-based logic for non-text files
        category = self._categorize_by_type_and_name(file_info)
        
        return {
            'file_name': file_info.name,
            'file_path': file_info.path,
            'file_type': file_info.file_type.value,
            'summary': f"{file_info.file_type.value} file: {file_info.name}",
            'category': category,
            'topics': [file_info.file_type.value],
            'importance': 'medium',
            'suggested_folder': self._suggest_folder_by_type(file_info.file_type),
            'confidence': 0.6
        }
    
    def _categorize_by_type_and_name(self, file_info: FileInfo) -> str:
        """Categorize file by type and name patterns"""
        
        name_lower = file_info.name.lower()
        
        # Work-related patterns
        if any(word in name_lower for word in ['meeting', 'project', 'report', 'presentation']):
            return 'Work'
        
        # Personal patterns
        if any(word in name_lower for word in ['personal', 'family', 'vacation', 'recipe']):
            return 'Personal'
        
        # Financial patterns
        if any(word in name_lower for word in ['invoice', 'receipt', 'tax', 'budget']):
            return 'Financial'
        
        # Educational patterns
        if any(word in name_lower for word in ['course', 'tutorial', 'study', 'notes']):
            return 'Educational'
        
        # By file type
        if file_info.file_type == FileType.IMAGE:
            return 'Media'
        elif file_info.file_type == FileType.VIDEO:
            return 'Media'
        elif file_info.file_type == FileType.AUDIO:
            return 'Media'
        elif file_info.file_type == FileType.CODE:
            return 'Technical'
        else:
            return 'General'
    
    def _suggest_folder_by_type(self, file_type: FileType) -> str:
        """Suggest folder based on file type"""
        
        if file_type == FileType.DOCUMENT:
            return 'Documents'
        elif file_type == FileType.IMAGE:
            return 'Images'
        elif file_type == FileType.VIDEO:
            return 'Videos'
        elif file_type == FileType.AUDIO:
            return 'Audio'
        elif file_type == FileType.CODE:
            return 'Code'
        elif file_type == FileType.ARCHIVE:
            return 'Archives'
        else:
            return 'Other'
    
    def _read_file_content(self, file_info: FileInfo) -> str:
        """Read file content for analysis"""
        
        # Skip large files
        if file_info.size > self.config.max_file_size_kb * 1024:
            logger.debug(f"Skipping large file: {file_info.name} ({file_info.size_mb:.1f} MB)")
            return ""
        
        try:
            if file_info.file_type == FileType.DOCUMENT:
                if file_info.extension.lower() == '.pdf':
                    return self._read_pdf_content(file_info.path)
                else:
                    return self._read_text_file(file_info.path)
            elif file_info.file_type == FileType.CODE:
                return self._read_text_file(file_info.path)
            else:
                return ""
        except Exception as e:
            logger.warning(f"Cannot read file {file_info.name}: {e}")
            return ""
    
    def _read_text_file(self, file_path: str) -> str:
        """Read plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(5000)  # Limit to 5KB
        except Exception:
            return ""
    
    def _read_pdf_content(self, file_path: str) -> str:
        """Read PDF content (requires PyPDF2)"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages[:3]:  # First 3 pages only
                    text += page.extract_text()
                return text[:5000]  # Limit to 5KB
        except ImportError:
            logger.debug("PyPDF2 not available for PDF reading")
            return ""
        except Exception:
            return ""
    
    def _get_cache_key(self, file_info: FileInfo) -> str:
        """Generate cache key for file"""
        content_hash = hashlib.md5(
            f"{file_info.path}:{file_info.modified_at}:{file_info.size}".encode()
        ).hexdigest()
        return f"llm_analysis:{content_hash}"
    
    def _create_fallback_structure(self, categories: Dict[str, List]) -> Dict[str, Any]:
        """Create fallback folder structure"""
        
        directories = []
        for category in categories.keys():
            if category.lower() == 'work':
                directories.extend(['Work/Projects', 'Work/Documents'])
            elif category.lower() == 'personal':
                directories.extend(['Personal/Documents', 'Personal/Photos'])
            elif category.lower() == 'financial':
                directories.append('Finance')
            elif category.lower() == 'educational':
                directories.append('Education')
            else:
                directories.append(f'Documents/{category}')
        
        return {
            'directories': directories,
            'reasoning': 'Fallback structure based on detected categories'
        }