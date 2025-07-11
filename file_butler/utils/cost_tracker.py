# file_butler/utils/cost_tracker.py
import json
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class UsageRecord:
    """Record of LLM usage"""
    timestamp: float
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    operation: str = "organization"

class CostTracker:
    """Track and manage LLM API costs"""
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        'openai': {
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        },
        'anthropic': {
            'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
            'claude-3-opus': {'input': 0.015, 'output': 0.075},
        }
    }
    
    def __init__(self, tracking_file: str = ".file_butler_cache/cost_tracking.json"):
        self.tracking_file = Path(tracking_file)
        self.tracking_file.parent.mkdir(exist_ok=True)
        self.usage_records: List[UsageRecord] = []
        self.load_tracking_data()
    
    def add_openai_usage(self, model: str, prompt_tokens: int, completion_tokens: int, operation: str = "organization"):
        """Add OpenAI usage record"""
        cost = self._calculate_openai_cost(model, prompt_tokens, completion_tokens)
        
        record = UsageRecord(
            timestamp=time.time(),
            provider='openai',
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            operation=operation
        )
        
        self.usage_records.append(record)
        self.save_tracking_data()
        
        logger.info(f"OpenAI usage: {prompt_tokens + completion_tokens} tokens, ${cost:.4f}")
    
    def add_anthropic_usage(self, model: str, input_tokens: int, output_tokens: int, operation: str = "organization"):
        """Add Anthropic usage record"""
        cost = self._calculate_anthropic_cost(model, input_tokens, output_tokens)
        
        record = UsageRecord(
            timestamp=time.time(),
            provider='anthropic',
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            cost_usd=cost,
            operation=operation
        )
        
        self.usage_records.append(record)
        self.save_tracking_data()
        
        logger.info(f"Anthropic usage: {input_tokens + output_tokens} tokens, ${cost:.4f}")
    
    def _calculate_openai_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate OpenAI cost"""
        if model not in self.PRICING['openai']:
            logger.warning(f"Unknown OpenAI model: {model}, using gpt-3.5-turbo pricing")
            model = 'gpt-3.5-turbo'
        
        pricing = self.PRICING['openai'][model]
        input_cost = (prompt_tokens / 1000) * pricing['input']
        output_cost = (completion_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost
    
    def _calculate_anthropic_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate Anthropic cost"""
        if model not in self.PRICING['anthropic']:
            logger.warning(f"Unknown Anthropic model: {model}, using claude-3-haiku pricing")
            model = 'claude-3-haiku'
        
        pricing = self.PRICING['anthropic'][model]
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost
    
    @property
    def total_cost_usd(self) -> float:
        """Get total cost across all usage"""
        return sum(record.cost_usd for record in self.usage_records)
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used"""
        return sum(record.prompt_tokens + record.completion_tokens for record in self.usage_records)
    
    def get_daily_costs(self, days: int = 30) -> Dict[str, float]:
        """Get daily costs for the last N days"""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        daily_costs = {}
        for record in self.usage_records:
            if record.timestamp > cutoff_time:
                date_str = time.strftime('%Y-%m-%d', time.localtime(record.timestamp))
                daily_costs[date_str] = daily_costs.get(date_str, 0) + record.cost_usd
        
        return daily_costs
    
    def get_model_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get cost breakdown by provider and model"""
        breakdown = {}
        
        for record in self.usage_records:
            provider = record.provider
            model = record.model
            
            if provider not in breakdown:
                breakdown[provider] = {}
            if model not in breakdown[provider]:
                breakdown[provider][model] = {'cost': 0, 'tokens': 0, 'calls': 0}
            
            breakdown[provider][model]['cost'] += record.cost_usd
            breakdown[provider][model]['tokens'] += record.prompt_tokens + record.completion_tokens
            breakdown[provider][model]['calls'] += 1
        
        return breakdown
    
    def save_tracking_data(self):
        """Save tracking data to file"""
        try:
            data = {
                'usage_records': [asdict(record) for record in self.usage_records],
                'last_updated': time.time()
            }
            
            with open(self.tracking_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cost tracking data: {e}")
    
    def load_tracking_data(self):
        """Load tracking data from file"""
        try:
            if self.tracking_file.exists():
                with open(self.tracking_file, 'r') as f:
                    data = json.load(f)
                
                self.usage_records = [
                    UsageRecord(**record) for record in data.get('usage_records', [])
                ]
        except Exception as e:
            logger.error(f"Failed to load cost tracking data: {e}")
            self.usage_records = []
    
    def generate_report(self) -> str:
        """Generate a cost usage report"""
        total_cost = self.total_cost_usd
        total_tokens = self.total_tokens
        total_calls = len(self.usage_records)
        
        if total_calls == 0:
            return "No LLM usage recorded yet."
        
        # Recent usage
        recent_records = [r for r in self.usage_records if r.timestamp > time.time() - 7*24*3600]
        recent_cost = sum(r.cost_usd for r in recent_records)
        
        # Model breakdown
        model_breakdown = self.get_model_breakdown()
        
        report = f"""LLM Usage Report
================

Total Usage:
- Cost: ${total_cost:.4f}
- Tokens: {total_tokens:,}
- API Calls: {total_calls}

Last 7 Days:
- Cost: ${recent_cost:.4f}
- Calls: {len(recent_records)}

Model Breakdown:
"""
        
        for provider, models in model_breakdown.items():
            report += f"\n{provider.title()}:\n"
            for model, stats in models.items():
                report += f"  {model}: ${stats['cost']:.4f} ({stats['tokens']:,} tokens, {stats['calls']} calls)\n"
        
        return report