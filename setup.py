import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    description = f.read()

__version__ = "0.0.0"

REPO_NAME = "CancerDetectionMLOps"
AUTHOR_USER_NAME = "pparashar21"
SRC_REPO = "CancerClassification"
AUTHOR_EMAIL = "prasoonparashar21@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Python package for multi-class cancer classification",
    long_description=description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = {
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where = "src")
)

