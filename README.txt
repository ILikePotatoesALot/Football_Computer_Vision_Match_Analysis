How to start the program:
1. Create a new anaconda env (recommended)
2. Activate the anaconda env
3. Download the dependencies from the requirements.txt (pip install -r requirements.txt)
4. Setup the API keys in the project root, so in conda write:
set DEEPSEEK_API_KEY=sk__
set ELEVENLABS_API_KEY=__
5. go to the project directory (cd GAIT_Assignment_AustinGoh_S10266831F)
6. Run the streamlit (streamlit run app_restructured.py)


Possible Issues:
"Model not found": In tracking_utils.py, change the model location to the exact location of your model.
(e.g. MODEL = YOLO("C:/Users/Austi/Sem4/Sem4 GAIT/GAIT ASSG/football-analyzer/Initia/Football_Custom.pt"))

Same for streamlit, if it doesn't work run the exact location of the file using python:
(e.g. python -m streamlit run "C:\Users\Austi\Sem4\Sem4 GAIT\GAIT ASSG\football-analyzer\initia\app_restructured.py)

Directory:
analyzed_videos: My already analyzed videos and is there if you wish to see the end product without analysis and to also see my progress towards the end product

Test_Sample_Videos: A series of 30 second clips you can test on the model to see it's effectiveness.

app_restructured.py: The main streamlit code

Football_Custom.pt: The custom trained yolo model used in the code

prompts.py: A separate file that stores all the exact prompts used for the LLMs in app_restructured.py

requirements.txt: A set of required libraries and dependencies needed to run the code

tracking_utils.py: the util file that stores the YOLO and bytetrack model weights and controls the video analysis portion.