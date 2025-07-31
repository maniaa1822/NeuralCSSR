"""
Machine implementations for Neural CSSR.

This module provides various epsilon-machine implementations:
- Domain-specific machines (even process, golden mean, etc.)
- Generic enumerated machines
- Custom machine builders
"""

from .domain_specific import (
    EvenProcessMachine,
    AlternatingMachine,
    GoldenMeanMachine,
    create_domain_specific_machine
)

__all__ = [
    'EvenProcessMachine',
    'AlternatingMachine', 
    'GoldenMeanMachine',
    'create_domain_specific_machine'
]
