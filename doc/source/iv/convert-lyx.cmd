"C:\Program Files (x86)\LyX 2.2\bin\lyx" --force-overwrite --export latex estimators.lyx
pandoc -s estimators.tex -o estimators.rst
copy /Y estimators.rst estimators-pre.txt
del estimators.rst
python -c "open('estimators.txt','w').write(''.join(open('estimators-pre.txt').readlines()[3:]))"
