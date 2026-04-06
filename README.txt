To start off, this code is incomplete since I can't upload the custom YOLO model and the videos i used to analyze itself. 
The video dataset can be found here: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/20
The model training can be found 


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
(e.g. MODEL = YOLO("C:/football-analyzer/Initia/Football_Custom.pt"))

Same for streamlit, if it doesn't work run the exact location of the file using python:
(e.g. python -m streamlit run "C:\football-analyzer\initia\app_restructured.py)

Directory:
app_restructured.py: The main streamlit code

Football_Custom.pt: The custom trained yolo model used in the code (NOT HERE)

prompts.py: A separate file that stores all the exact prompts used for the LLMs in app_restructured.py

requirements.txt: A set of required libraries and dependencies needed to run the code

tracking_utils.py: the util file that stores the YOLO and bytetrack model weights and controls the video analysis portion.
