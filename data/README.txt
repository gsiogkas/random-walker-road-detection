Dimension of the compressed archive file: 224330461 Bytes.


> gunzip gtseq002836.tgz 
> tar -xvf gtseq002836.tar

In the directory gtseq you will now find the following files:

diplo000000-L.png
diplo000000-L.txt
diplo000000-R.png
....
diplo000864-L.png
diplo000864-L.txt
diplo000864-R.png
svscalib.ini
README.txt
timestamps.txt 



_ svscalib.ini is the stereo pair configuration computed by the SVS
calibration module

###########################################

_ every image pair is represented by 3 files:
  1_ diplo%06d-L.png : 15fps 320x240 LEFT color image
  2_ diplo%06d-R.png : 15fps 320x240 RIGHT color image
  3_ diplo%06d-L.txt : segmentation ground truth with respect to the
  left image

  where %06d = 000000 - 000864

###########################################

_ every line of the diplo%06d-L.txt files represents a polygon and has
the following structure:

<type> <N> x1 y1 x2 y2 ... xN yN

where:

<type> can be 'road' (an ideal road region) or 'occl' (a region that
overlaps the road occluding a part of it)

<N> is the number of vertices of the polygon

xi yi, i=1..N are the vertices, with xi and yi normalized into the
0.0-1.0 range

###########################################

_ for a particular image, if Ri (i=1..Nr) are the road regions and Cj
(j=1..Nc) are the occluding regions, the visible road region is 
V = Ui{Ri} \ Uj{Cj}

###########################################

_ the 865 images are taken from 5 different subsequences:
  1:   0 - 450 (450 frames)
  2: 451 - 601 (100 frames)
  3: 602 - 702 (100 frames)
  4: 703 - 763 ( 60 frames)
  5: 764 - 864 (100 frames)

###########################################

_ the file timestamps.txt contains the capture timestamps of some
frames in the form:
<frame number> <timestamp>
       where <timestamp> has the following strftime format :
       "%Y/%m/%d %H:%M:%S.%Q", where %Q are milliseconds (000 - 999)


---

Please cite:
M. Zanin, S. Messelodi, C.M. Modena
DIPLODOC road stereo sequence,
FBK-irst Technical Report #164010, April 2013 