#!/usr/bin/env bash

echo "Checkout pages"
git checkout gh-pages
echo "Remove devel"
rm -rf devel
echo "Make a new devel"
mkdir devel
echo "Move docs"
mv doc/build/html/* devel/
echo "Add devel"
git add devel/.
echo "Change remove"
git remote set-url origin https://bashtage:"${GH_PAGES_TOKEN}"@github.com/bashtage/linearmodels.git
echo "Github Actions doc build after commit ${GITHUB_REF##*/}"
git commit -a -m "Github Actions doc build after commit ${GITHUB_REF##*/}"
echo "Push"
git push -f



