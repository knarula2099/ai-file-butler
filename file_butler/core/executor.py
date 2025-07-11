# file_butler/core/executor.py
import os
import shutil
from pathlib import Path
from typing import List, Optional, Callable
import logging
import time

from .models import Action, ActionType, OrganizationPlan

logger = logging.getLogger(__name__)

class ActionExecutor:
    """Executes file operations safely with dry-run support"""
    
    def __init__(self, 
                 dry_run: bool = True,
                 backup_mode: bool = False,
                 progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """
        Args:
            dry_run: If True, only simulate actions without executing them
            backup_mode: If True, create backups before destructive operations
            progress_callback: Function called with (current, total, action_description)
        """
        self.dry_run = dry_run
        self.backup_mode = backup_mode
        self.progress_callback = progress_callback
        
        # Statistics
        self.executed_actions = 0
        self.failed_actions = 0
        self.execution_log = []
    
    def execute_plan(self, plan: OrganizationPlan) -> 'ExecutionResult':
        """Execute an organization plan"""
        
        start_time = time.time()
        self.executed_actions = 0
        self.failed_actions = 0
        self.execution_log = []
        
        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Executing plan with {len(plan.actions)} actions")
        
        # Group actions by type for optimal execution order
        ordered_actions = self._order_actions(plan.actions)
        
        for i, action in enumerate(ordered_actions):
            try:
                self._execute_action(action)
                self.executed_actions += 1
                
                if self.progress_callback:
                    self.progress_callback(i + 1, len(ordered_actions), str(action))
                    
            except Exception as e:
                self.failed_actions += 1
                error_msg = f"Failed to execute {action}: {e}"
                self.execution_log.append(f"ERROR: {error_msg}")
                logger.error(error_msg)
        
        execution_time = time.time() - start_time
        
        return ExecutionResult(
            plan=plan,
            executed_actions=self.executed_actions,
            failed_actions=self.failed_actions,
            execution_time=execution_time,
            execution_log=self.execution_log.copy(),
            dry_run=self.dry_run
        )
    
    def _order_actions(self, actions: List[Action]) -> List[Action]:
        """Order actions for safe execution (directories first, deletes last)"""
        
        create_actions = [a for a in actions if a.action_type == ActionType.CREATE_DIR]
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE]
        copy_actions = [a for a in actions if a.action_type == ActionType.COPY]
        rename_actions = [a for a in actions if a.action_type == ActionType.RENAME]
        delete_actions = [a for a in actions if a.action_type == ActionType.DELETE]
        
        # Safe execution order
        return create_actions + copy_actions + move_actions + rename_actions + delete_actions
    
    def _execute_action(self, action: Action):
        """Execute a single action"""
        
        if action.action_type == ActionType.CREATE_DIR:
            self._create_directory(action)
        elif action.action_type == ActionType.MOVE:
            self._move_file(action)
        elif action.action_type == ActionType.COPY:
            self._copy_file(action)
        elif action.action_type == ActionType.DELETE:
            self._delete_file(action)
        elif action.action_type == ActionType.RENAME:
            self._rename_file(action)
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
    
    def _create_directory(self, action: Action):
        """Create a directory"""
        target_path = Path(action.destination)
        
        if self.dry_run:
            log_msg = f"[DRY RUN] Would create directory: {target_path}"
            print(log_msg)
            self.execution_log.append(log_msg)
        else:
            try:
                target_path.mkdir(parents=True, exist_ok=True)
                log_msg = f"Created directory: {target_path}"
                self.execution_log.append(log_msg)
                logger.info(log_msg)
            except Exception as e:
                raise Exception(f"Cannot create directory {target_path}: {e}")
    
    def _move_file(self, action: Action):
        """Move a file"""
        source_path = Path(action.source)
        target_path = Path(action.destination)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")
        
        # Ensure target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.dry_run:
            log_msg = f"[DRY RUN] Would move: {source_path.name} → {target_path}"
            print(log_msg)
            self.execution_log.append(log_msg)
        else:
            try:
                # Handle file conflicts
                if target_path.exists():
                    target_path = self._resolve_conflict(target_path)
                
                shutil.move(str(source_path), str(target_path))
                log_msg = f"Moved: {source_path.name} → {target_path}"
                self.execution_log.append(log_msg)
                logger.info(log_msg)
            except Exception as e:
                raise Exception(f"Cannot move {source_path} to {target_path}: {e}")
    
    def _copy_file(self, action: Action):
        """Copy a file"""
        source_path = Path(action.source)
        target_path = Path(action.destination)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")
        
        # Ensure target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.dry_run:
            log_msg = f"[DRY RUN] Would copy: {source_path.name} → {target_path}"
            print(log_msg)
            self.execution_log.append(log_msg)
        else:
            try:
                # Handle file conflicts
                if target_path.exists():
                    target_path = self._resolve_conflict(target_path)
                
                shutil.copy2(str(source_path), str(target_path))
                log_msg = f"Copied: {source_path.name} → {target_path}"
                self.execution_log.append(log_msg)
                logger.info(log_msg)
            except Exception as e:
                raise Exception(f"Cannot copy {source_path} to {target_path}: {e}")
    
    def _delete_file(self, action: Action):
        """Delete a file"""
        source_path = Path(action.source)
        
        if not source_path.exists():
            logger.warning(f"File to delete does not exist: {source_path}")
            return
        
        if self.dry_run:
            log_msg = f"[DRY RUN] Would delete: {source_path.name} ({action.reason})"
            print(log_msg)
            self.execution_log.append(log_msg)
        else:
            try:
                if self.backup_mode:
                    self._backup_file(source_path)
                
                source_path.unlink()
                log_msg = f"Deleted: {source_path.name} ({action.reason})"
                self.execution_log.append(log_msg)
                logger.info(log_msg)
            except Exception as e:
                raise Exception(f"Cannot delete {source_path}: {e}")
    
    def _rename_file(self, action: Action):
        """Rename a file"""
        source_path = Path(action.source)
        target_path = Path(action.destination)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")
        
        if self.dry_run:
            log_msg = f"[DRY RUN] Would rename: {source_path.name} → {target_path.name}"
            print(log_msg)
            self.execution_log.append(log_msg)
        else:
            try:
                # Handle file conflicts
                if target_path.exists():
                    target_path = self._resolve_conflict(target_path)
                
                source_path.rename(target_path)
                log_msg = f"Renamed: {source_path.name} → {target_path.name}"
                self.execution_log.append(log_msg)
                logger.info(log_msg)
            except Exception as e:
                raise Exception(f"Cannot rename {source_path} to {target_path}: {e}")
    
    def _resolve_conflict(self, target_path: Path) -> Path:
        """Resolve file name conflicts by adding a suffix"""
        base = target_path.stem
        suffix = target_path.suffix
        parent = target_path.parent
        
        counter = 1
        while True:
            new_name = f"{base}_{counter}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1
    
    def _backup_file(self, file_path: Path):
        """Create a backup of a file before deletion"""
        backup_dir = file_path.parent / ".file_butler_backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / f"{file_path.name}.backup"
        shutil.copy2(str(file_path), str(backup_path))
        logger.info(f"Created backup: {backup_path}")

class ExecutionResult:
    """Result of executing an organization plan"""
    
    def __init__(self, 
                 plan: OrganizationPlan,
                 executed_actions: int,
                 failed_actions: int,
                 execution_time: float,
                 execution_log: List[str],
                 dry_run: bool):
        self.plan = plan
        self.executed_actions = executed_actions
        self.failed_actions = failed_actions
        self.execution_time = execution_time
        self.execution_log = execution_log
        self.dry_run = dry_run
    
    @property
    def success_rate(self) -> float:
        """Percentage of successfully executed actions"""
        total = self.executed_actions + self.failed_actions
        if total == 0:
            return 100.0
        return (self.executed_actions / total) * 100
    
    def summary(self) -> str:
        """Get a summary of the execution"""
        mode = "DRY RUN" if self.dry_run else "EXECUTION"
        return (f"{mode} Summary:\n"
                f"  Actions executed: {self.executed_actions}\n"
                f"  Actions failed: {self.failed_actions}\n"
                f"  Success rate: {self.success_rate:.1f}%\n"
                f"  Execution time: {self.execution_time:.2f}s")