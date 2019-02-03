======================================================================  
I made this code for letting HTM learners like myself   
to see connections between HTM school videos and NuPIC code which is based on HTM theory  
  
You can find HTM school videos in  
https://www.youtube.com/playlist?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9  
  
You can find original NuPIC code in  
https://github.com/numenta/nupic  
  
======================================================================  
Codes are originated from   
https://github.com/psdyer/NuPIC-Algorithm-API-Example  
  
And I added   
"connection between HTM school video and NuPIC code",  
"my explanatory comments"  
  
You can read detailed explanations in base code  
https://github.com/psdyer/NuPIC-Algorithm-API-Example/blob/master/algorithm_api_example.ipynb  
  
======================================================================  
Appending c before variable name is my commenting style like following  
c timeOfDayEncoder: you create date encoder which will encode timeOfDay input data  
  
I use this because I can search explanatory comments on any variables easily.  
You will be able to find comment for timeOfDayEncoder   
by pressing CTRL+F with searching c timeOfDayEncoder  
  
I also used this way for this code.  
  
======================================================================  
There is "List" in main file, on which you can have overview of all parts like this  
  
List  
SDR Capacity & Comparison (Episode 2), 3:00  
Scalar Encoding (Episode 5), 1:46  
Scalar Encoding (Episode 5), 8:33  
Scalar Encoding (Episode 5), 9:43  
...  
  
======================================================================  
Each part has @ mark like following.  
@ Datetime Encoding (Episode 6), 3:16  
  
So if you search "@ ",  
you can jump to all of parts easily.  
  
======================================================================  
How to use  
  
1. Install NuPIC according to   
https://github.com/numenta/nupic#installing-nupic  
2. Download one_hot_gym_data.csv contained in this repository  
and set its path into code by searching one_hot_gym_data (variable name is _INPUT_FILE_PATH) in code  
3. Run code  
python connection_between_video_and_code.py  
