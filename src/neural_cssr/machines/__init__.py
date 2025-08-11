"""
Machine implementations for Neural CSSR.

This module provides various epsilon-machine implementations:
- Domain-specific machines (even process, golden mean, etc.)
- Generic enumerated machines
- Custom machine builders
"""

from .domain_specific import (
    BiasedCoinMachine,
    AlternatingMachine,
    GoldenMeanMachine,
    create_domain_specific_machine
)

__all__ = [
    'BiasedCoinMachine',
    'AlternatingMachine', 
    'GoldenMeanMachine',
    'create_domain_specific_machine'
]
