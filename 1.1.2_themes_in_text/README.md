# Python Libraries

```bash
pip3 install beautifulsoup4 lxml html5lib
```

# Running

```bash
# 1: scrape faculty data and save to data/faculty
python3 1_extract_mit_professors.py

# 2: using faculty data, scrape all abstracts and save to data/articles/[faculty name]
python3 2_search_archives.py

# 3: using abstracts, run vectorize/count the abstracts, run LDA and plot
python3 3_themes_in_text.py
```