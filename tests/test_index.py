import os
from bs4 import BeautifulSoup

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_html_exists():
    assert os.path.exists("index.html")

def test_html_loads():
    with open("index.html", "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    # Basic check: required HTML container exists
    chart = soup.find(id="chartContainer")
    assert chart is not None
