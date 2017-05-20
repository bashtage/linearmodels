"C:\Program Files (x86)\LyX 2.2\bin\lyx" --force-overwrite --export latex mathematical-detail.lyx
pandoc -s mathematical-detail.tex -o mathematical-detail.rst
copy /Y mathematical-detail.rst mathematical-detail-pre.txt
del mathematical-detail.rst
python -c "open('mathematical-detail.txt','w').write(''.join(open('mathematical-detail-pre.txt').readlines()[3:]))"
del mathematical-detail-pre.txt
del mathematical-detail.tex