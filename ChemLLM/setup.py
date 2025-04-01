from setuptools import setup, find_packages

setup(
    name="ChemLLM",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A chemistry-focused LLM project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ChemLLM",
    packages=find_packages(include=["ChemLLM", "ChemLLM.*"]),
    install_requires=[
        # Keep this empty if using pyproject.toml for dependencies
    ],
    python_requires='>=3.8',
)