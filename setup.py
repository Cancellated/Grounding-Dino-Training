"""
Setup script for Grounding DINO Rust extensions
"""
from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="groundingdino-rust",
    version="0.1.0",
    rust_extensions=[
        RustExtension(
            "groundingdino_rust",
            "rust/Cargo.toml",
            binding=Binding.PyO3
        )
    ],
    setup_requires=["setuptools-rust>=1.0.0", "wheel"],
    zip_safe=False,
    packages=[],
    include_package_data=False,
)
