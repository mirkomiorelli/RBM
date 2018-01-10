ffmpeg -r 25 -i images/img%05d.png -vb 50M -y nh100.avi
ffmpeg -i nh100.avi -vb 20M -y nh20_banner.gif
