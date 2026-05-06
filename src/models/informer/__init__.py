"""
src/models/informer/__init__.py
Re-exports the Informer and InformerStack classes with fixed internal imports.
"""
# Fix import paths so the package works standalone (no sys.path hacks needed)
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from .model import Informer, InformerStack

__all__ = ["Informer", "InformerStack"]
