"C:\Program Files\LyX 2.3\bin\LyX" --force-overwrite --export latex mathematical-detail.lyx
pandoc -s mathematical-detail.tex -o mathematical-detail.rst
copy /Y mathematical-detail.rst mathematical-detail.txt
del mathematical-detail.rst
rem python -c "open('mathematical-detail.txt','w').write(''.join(open('mathematical-detail-pre.txt').readlines()[3:]))"
rem del mathematical-detail-pre.txt
