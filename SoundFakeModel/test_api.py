import requests


def test_upload_file(path):
    # Create a sample file (you can use a temporary file)
    # ...

    files = {'audio': open(path, 'rb')}  # Use 'rb' for binary mode
    response = requests.post('http://127.0.0.1:8080/upload', files=files)
    print("returned")
    print(response.text)

test_upload_file('/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/DeepfakeSoundFiles/MarioDeepfake.mp3')

