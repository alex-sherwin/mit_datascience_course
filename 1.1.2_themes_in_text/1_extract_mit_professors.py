#!/usr/bin/env python3

from bs4 import BeautifulSoup
import requests
import pathlib
import json


mit_labs = set(['CSAIL', 'LIDS', 'MTL', 'RLE'])

page = requests.get('https://www.eecs.mit.edu/people/faculty-advisors')
soup = BeautifulSoup(page.content, features='html.parser')

def extractFacultyData(facultyNode):
    nameAnchors = facultyNode.select('div.views-field-title a')
    # if a name wasn't found, quit
    if len(nameAnchors) != 1:
      return None
    name = nameAnchors[0].get_text().strip()
    labAnchors = facultyNode.select('div.views-field-term-node-tid a')
    labs = list(
        map(lambda anchor: anchor.get_text().strip(), labAnchors)
    )
    # if they don't have one of the mit labs, quit
    if not bool(set(labs) & mit_labs):
      return None
    # filter down to known labs only
    labs = [i for i in labs if bool(set([i]) & mit_labs)]

    return {'name': name, 'labs': labs}


facultyListItems = soup.select('ul.faculty-list li')
facultyDataWithNone = list(
  map(lambda node: extractFacultyData(node), facultyListItems)
)
# filter out None values (missing faculty name or doesn't have a mit_lab)
facultyData = [i for i in facultyDataWithNone if i] 

pathlib.Path('data').mkdir(parents=True, exist_ok=True)
outFile = open('data/faculty', 'w')

outFile.write(json.dumps(facultyData, indent=2))
