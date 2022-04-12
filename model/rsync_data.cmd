gsname='gs0317'
echo "RSYNC GRIDSEARCH DATA FROM:"
echo ${gsname}
echo

rsync -r -vam --progress --update abeukers@scotty.princeton.edu:/jukebox/norman/abeukers/wd/CSW/cswsem0621/data/${gsname}/* /Users/abeukers/wd/csw/cswsem0621/data/${gsname}
