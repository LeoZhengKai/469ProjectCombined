"""
MIPROv2 optimizer implementation.

This module provides a MIPROv2 optimizer with configuration
options for multi-prompt optimization.
"""

import dspy
from typing import Any, Dict, List, Optional
from pathlib import Path


class MIPROv2Optimizer:
    """
    MIPROv2 optimizer with configurable parameters.
    
    This optimizer uses multi-prompt optimization to improve
    performance across different prompts and configurations.
    """
    
    def __init__(
        self,
        num_candidates: int = 10,
        init_temperature: float = 1.0,
        verbose: bool = False,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        max_rounds: int = 1,
        num_threads: int = 6,
        metric: Optional[Any] = None
    ):
        """
        Initialize the MIPROv2 optimizer.
        
        Args:
            num_candidates: Number of candidate prompts to generate
            init_temperature: Initial temperature for generation
            verbose: Whether to print verbose output
            max_bootstrapped_demos: Maximum number of bootstrapped demonstrations
            max_labeled_demos: Maximum number of labeled demonstrations
            max_rounds: Maximum number of optimization rounds
            num_threads: Number of threads for parallel processing
            metric: Evaluation metric to optimize
        """
        self.num_candidates = num_candidates
        self.init_temperature = init_temperature
        self.verbose = verbose
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.num_threads = num_threads
        self.metric = metric
        
        # Initialize the optimizer
        self._optimizer = None
        self._initialize_optimizer()
    
    def _initialize_optimizer(self) -> None:
        """Initialize the DSPy MIPROv2 optimizer."""
        try:
            self._optimizer = dspy.MIPROv2(
                num_candidates=self.num_candidates,
                init_temperature=self.init_temperature,
                verbose=self.verbose,
                max_bootstrapped_demos=self.max_bootstrapped_demos,
                max_labeled_demos=self.max_labeled_demos,
                max_rounds=self.max_rounds,
                num_threads=self.num_threads,
                metric=self.metric
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MIPROv2 optimizer: {e}")
    
    def optimize(self, program: dspy.Module, trainset: List[Any]) -> dspy.Module:
        """
        Optimize a DSPy program using MIPROv2.
        
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
            "type": "MIPROv2",
            "num_candidates": self.num_candidates,
            "init_temperature": self.init_temperature,
            "verbose": self.verbose,
            "max_bootstrapped_demos": self.max_bootstrapped_demos,
            "max_labeled_demos": self.max_labeled_demos,
            "max_rounds": self.max_rounds,
            "num_threads": self.num_threads,
            "metric": str(self.metric) if self.metric else None,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MIPROv2Optimizer":
        """
        Create optimizer from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured optimizer instance
        """
        return cls(
            num_candidates=config.get("num_candidates", 10),
            init_temperature=config.get("init_temperature", 1.0),
            verbose=config.get("verbose", False),
            max_bootstrapped_demos=config.get("max_bootstrapped_demos", 4),
            max_labeled_demos=config.get("max_labeled_demos", 16),
            max_rounds=config.get("max_rounds", 1),
            num_threads=config.get("num_threads", 6),
            metric=config.get("metric")
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
    def load_config(cls, file_path: Path) -> "MIPROv2Optimizer":
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
