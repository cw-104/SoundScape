from sound_scape.backend.Models import xlsr, whisper_specrnet

path = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Example Files/DeepfakeSoundFiles/BillieEilishDeepfake.mp3"

model = xlsr()
# model = whisper_specrnet()
print(model.evaluate(path))