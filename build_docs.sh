#!/bin/bash

# we use pdoc3
pip install 'git+https://github.com/pdoc3/pdoc.git#egg=pdoc3'

pdoc --html --output-dir docs/ -c show_type_annotations=True ./pbl

# publish docs if we run this on travis with master
if [ "$TRAVIS_BRANCH" = "master" -a "$TRAVIS_PULL_REQUEST" = "false" ]; then
	pip install ghp-import
	ghp-import -n docs/pbl
	git push -qf https://${GH_TOKEN}@github.com/${TRAVIS_REPO_SLUG}.git gh-pages
fi;
