"""
BootstrapFewShot optimizer implementation.

This module provides a BootstrapFewShot optimizer with configuration
options for few-shot learning and bootstrap sampling.
"""

import dspy
from typing import Any, Dict, List, Optional
from pathlib import Path


class BootstrapFewShotOptimizer:
    """
    BootstrapFewShot optimizer with configurable parameters.
    
    This optimizer uses bootstrap sampling to select few-shot examples
    for prompt optimization. It's particularly effective for tasks
    with limited training data.
    """
    
    def __init__(
        self,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        max_rounds: int = 1,
        teacher_settings: Optional[Dict[str, Any]] = None,
        student_settings: Optional[Dict[str, Any]] = None,
        metric: Optional[Any] = None,
        num_candidate_programs: int = 16,
        num_threads: int = 6
    ):
        """
        Initialize the BootstrapFewShot optimizer.
        
        Args:
            max_bootstrapped_demos: Maximum number of bootstrapped demonstrations
            max_labeled_demos: Maximum number of labeled demonstrations
            max_rounds: Maximum number of optimization rounds
            teacher_settings: Settings for the teacher model
            student_settings: Settings for the student model
            metric: Evaluation metric to optimize
            num_candidate_programs: Number of candidate programs to generate
            num_threads: Number of threads for parallel processing
        """
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.teacher_settings = teacher_settings or {}
        self.student_settings = student_settings or {}
        self.metric = metric
        self.num_candidate_programs = num_candidate_programs
        self.num_threads = num_threads
        
        # Initialize the optimizer
        self._optimizer = None
        self._initialize_optimizer()
    
    def _initialize_optimizer(self) -> None:
        """Initialize the DSPy BootstrapFewShot optimizer."""
        try:
            self._optimizer = dspy.BootstrapFewShot(
                max_bootstrapped_demos=self.max_bootstrapped_demos,
                max_labeled_demos=self.max_labeled_demos,
                max_rounds=self.max_rounds,
                teacher_settings=self.teacher_settings,
                student_settings=self.student_settings,
                metric=self.metric,
                num_candidate_programs=self.num_candidate_programs,
                num_threads=self.num_threads
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BootstrapFewShot optimizer: {e}")
    
    def optimize(self, program: dspy.Module, trainset: List[Any]) -> dspy.Module:
        """
        Optimize a DSPy program using BootstrapFewShot.
        
        Args:
            program: DSPy program to optimize
            trainset: Training dataset
            
        Returns:
            Optimized DSPy program
        """
        if self._optimizer is None:
            raise RuntimeError("Optimizer not initialized")
        
        try:
            optimized_program = self._optimizer.compile(program, trainset=trainset)
            return optimized_program
        except Exception as e:
            raise RuntimeError(f"Failed to optimize program: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current optimizer configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            "type": "BootstrapFewShot",
            "max_bootstrapped_demos": self.max_bootstrapped_demos,
            "max_labeled_demos": self.max_labeled_demos,
            "max_rounds": self.max_rounds,
            "teacher_settings": self.teacher_settings,
            "student_settings": self.student_settings,
            "metric": str(self.metric) if self.metric else None,
            "num_candidate_programs": self.num_candidate_programs,
            "num_threads": self.num_threads,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BootstrapFewShotOptimizer":
        """
        Create optimizer from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured optimizer instance
        """
        return cls(
            max_bootstrapped_demos=config.get("max_bootstrapped_demos", 4),
            max_labeled_demos=config.get("max_labeled_demos", 16),
            max_rounds=config.get("max_rounds", 1),
            teacher_settings=config.get("teacher_settings", {}),
            student_settings=config.get("student_settings", {}),
            metric=config.get("metric"),
            num_candidate_programs=config.get("num_candidate_programs", 16),
            num_threads=config.get("num_threads", 6)
        )
    
    def save_config(self, file_path: Path) -> None:
        """
        Save optimizer configuration to file.
        
        Args:
            file_path: Path to save configuration
        """
        config = self.get_config()
        import json
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_config(cls, file_path: Path) -> "BootstrapFewShotOptimizer":
        """
        Load optimizer configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configured optimizer instance
        """
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return cls.from_config(config)
