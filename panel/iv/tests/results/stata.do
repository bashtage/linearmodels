use http://www.stata-press.com/data/r13/hsng, clear

ivregress 2sls rent pcturban (hsngval = faminc i.region)
estout using c:\git\panel\panel\iv\tests\results\stata-iv2sls-unadjusted.txt, cells(b(fmt(%13.12g)) t(fmt(%13.12g))) stats(r2 r2_a chi2 p mss rss, fmt(%13.12g)) unstack replace
matrix V = e(V)
file open myfile using "c:\git\panel\panel\iv\tests\results\stata-iv2sls-unadjusted.txt", write append
file write myfile  "************************************\n"
file close myfile
estout matrix(V, fmt(%13.12g)) using c:\git\panel\panel\iv\tests\results\stata-iv2sls-unadjusted.txt, append


ivregress 2sls rent pcturban (hsngval = faminc i.region), vce(robust)
estout using c:\git\panel\panel\iv\tests\results\stata-iv2sls-robust.txt, cells(b(fmt(%13.12g)) t(fmt(%13.12g))) stats(r2 r2_a chi2 p mss rss, fmt(%13.12g)) unstack replace
matrix V = e(V)
file open myfile using "c:\git\panel\panel\iv\tests\results\stata-iv2sls-robust.txt", write append
file write myfile  "************************************\n"
file close myfile
estout matrix(V, fmt(%13.12g)) using c:\git\panel\panel\iv\tests\results\stata-iv2sls-robust.txt, append

ivregress gmm rent pcturban (hsngval = faminc i.region), wmatrix(unadjusted)
estout using c:\git\panel\panel\iv\tests\results\stata-ivgmm-unadjusted.txt, cells(b(fmt(%13.12g)) t(fmt(%13.12g))) stats(r2 r2_a chi2 p mss rss, fmt(%13.12g)) unstack replace
matrix V = e(V)
file open myfile using "c:\git\panel\panel\iv\tests\results\stata-ivgmm-unadjusted.txt", write append
file write myfile  "************************************\n"
file close myfile
estout matrix(V, fmt(%13.12g)) using c:\git\panel\panel\iv\tests\results\stata-ivgmm-unadjusted.txt, append

ivregress gmm rent pcturban (hsngval = faminc i.region), wmatrix(robust)
estout using c:\git\panel\panel\iv\tests\results\stata-ivgmm-robust.txt, cells(b(fmt(%13.12g)) t(fmt(%13.12g))) stats(r2 r2_a chi2 p mss rss, fmt(%13.12g)) unstack replace
matrix V = e(V)
file open myfile using "c:\git\panel\panel\iv\tests\results\stata-ivgmm-robust.txt", write append
file write myfile  "************************************\n"
file close myfile
estout matrix(V, fmt(%13.12g)) using c:\git\panel\panel\iv\tests\results\stata-ivgmm-robust.txt, append



