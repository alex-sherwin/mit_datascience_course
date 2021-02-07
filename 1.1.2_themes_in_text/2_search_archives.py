#!/usr/bin/env python3

from bs4 import BeautifulSoup
import requests
import pathlib
import urllib
import json


def searchUrl(name):
    return '{prefix}{author}{suffix}'.format(
        prefix="https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&",
        author=urllib.parse.urlencode({'terms-0-term': name}),
        suffix="&terms-0-field=author&classification-physics_archives=all&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=200&order=-announced_date_first"
    )


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def extractSearchResultData(soup):
    archiveId = remove_prefix(
        # pull out the archive id
        soup.select('p.list-title a')[0].get_text().strip(),
        "arXiv:"  # strip off the arXiv: prefix
    )

    authors = list(map(lambda anchor: anchor.get_text(
    ).strip(), soup.select('p.authors a')))

    # facultyNames = list(map(lambda anchor: anchor.get_text().strip(), facultyAnchors))

    title = soup.select('p.title')[0].get_text().strip()
    abstract = soup.select('p.abstract')[0].get_text().strip()
    return {'id': archiveId, 'title': title,  'abstract': abstract, 'authors': authors}


def processFacultyName(name, facultyLabs):
    # make HTTP request for the data
    page = requests.get(searchUrl(name))
    # parse the HTML page
    soup = BeautifulSoup(page.content, features='html.parser')

    # use a CSS selector to get an array of search result <li> nodes
    searchResults = soup.select('li.arxiv-result')

    # for each search result <li> call extractSearchResultData to extract all article data
    articlesData = list(
        map(lambda x: extractSearchResultData(x), searchResults)
    )

    # wrap up scraped article data list next to author name in a dict
    byAuthorData = {'author': name, 'labs': facultyLabs, 'articles':  articlesData}

    outFile = open('data/articles/' + name, 'w')
    outFile.write(json.dumps(byAuthorData, indent=2))


# ensure a data/articles folder exists to hold our parsed article data
pathlib.Path('data/articles').mkdir(parents=True, exist_ok=True)

# read all faculty member names
facultyDatas = json.loads(open('data/faculty', 'r').read())

# for each faculty member, process (perform an arXiv search, scrape results, save as JSON to a local file)
for facultyData in facultyDatas:
    name = facultyData['name']
    print('Starting scraping for ' + name + '...')
    processFacultyName(name, facultyData['labs'])
    print('  Done scraping for ' + name)
