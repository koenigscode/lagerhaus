# Lagerhaus Feature Store

> **Note**: This package is a proof-of-concept and is not intended for production use.

This is a feature store integrating data cleaning, an artefact of our Bachelor's thesis _"Developing a Feature Store Library with Built-In Data Cleaning Support"_ by [Michael König](https://www.linkedin.com/in/koenig-michael/) and [Sofia Serbina](https://www.linkedin.com/in/sofia-serbina-2b7182264/).

## Usage

Since this package is more of a proof-of-concept, we decided not to publish it on PyPI.
Instead, you can install the package using pip:

```bash
pip install git+https://github.com/koenigscode/lagerhaus.git@main
```

If the package is already installed and you want to update it
(but the semver version hasn't changed), you have to uninstall the package first,
due to caching.

## Library Development

To work on the library itself, follow these steps:

### Setup

(Recommended: Use a virtual environment)

```bash
python3 -m venv venv
source ./venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

### Test / Example

To run the example, which includes a web-based UI, use streamlit:

```bash
python -m streamlit run tests/test.py
```
