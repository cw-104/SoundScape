# Top 50 artists in The United States that we support for vocal identification (All song samples contain no feautures):
# Taylor Swift
# Billie Eilish
# Kendrick Lamar
# The Weeknd
# Rihanna
# Ariana Grande
# Ed Sheeran
# Beyoncé
# Adele
# Justin Bieber
# Drake
# Lady Gaga
# Katy Perry
# Bruno Mars
# Post Malone
# Miley Cyrus
# SZA
# Shakira
# Chris Brown
# Kanye West
# Lana Del Rey
# Demi Lovato
# J. Cole
# Travis Scott
# Gracie Abrams
# Lil Nas X
# Doja Cat
# Harry Styles
# Olivia Rodrigo
# Dua Lipa
# Bad Bunny
# Nicki Minaj
# Eminem
# Jay-Z
# Lil Wayne
# Cardi B
# Selena Gomez
# Camila Cabello
# Shawn Mendes
# Sabrina Carpenter
# Morgan Wallen
# Playboi Carti
# Chappell Roan
# Zach Bryan
# Hozier
# Luke Combs
# Frank Ocean
# Tyler, the Creator
# Tate McRae
# Noah Kahan

from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from collections import defaultdict
from pathlib import Path
from numpy import dot
from numpy.linalg import norm

# ignore warnings
import warnings

warnings.filterwarnings("ignore")


# File to analyze for speaker match
input_file_path = "/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ArianaGrande/Ariana Grande - Break Free (Official Video) ft. Zedd.mp3" # path to the audio file you want to analyze

wav_fpath = Path(input_file_path) # turns string into path cause thats what lib wants
wav = preprocess_wav(wav_fpath)


# REFRENCE AUIDO 2 array MUST BE IN MATCHING ORDER
# speaker_names = [speaker1, speaker2, ...]
# speaker_wavs = [speaker1_audio_path.mp3, speaker2_audio_path.mp3, ...]

# Names of the singers
speaker_names = [
"Adele1", "Adele2", "Adele3",
"ArianaGrande1", "ArianaGrande2", "ArianaGrande3",
"BadBunny1", "BadBunny2", "BadBunny3",
"Beyonce1", "Beyonce2", "Beyonce3",
"BillieEilish1", "BillieEilish2", "BillieEilish3",
"BrunoMars1", "BrunoMars2", "BrunoMars3",
"CamilaCabello1", "CamilaCabello2", "CamilaCabello3",
"CardiB1", "CardiB2", "CardiB3",
"ChappellRoan1", "ChappellRoan2", "ChappellRoan3",
"DemiLovato1", "DemiLovato2", "DemiLovato3",
"DojaCat1", "DojaCat2", "DojaCat3",
"Drake1", "Drake2", "Drake3",
"DuaLipa1", "DuaLipa2", "DuaLipa3",
"EdSheeran1", "EdSheeran2", "EdSheeran3",
"Eminem1", "Eminem2", "Eminem3",
"FrankOcean1", "FrankOcean2", "FrankOcean3",
"GracieAbrams1", "GracieAbrams2", "GracieAbrams3",
"HarryStyles1", "HarryStyles2", "HarryStyles3",
"Hozier1", "Hozier2", "Hozier3",
"JayZ1", "JayZ2", "JayZ3",
"Jcole1", "Jcole2", "Jcole3",
"JustinBieber1", "JustinBieber2", "JustinBieber3",
"KatyPerry1", "KatyPerry2", "KatyPerry3",
"KendrickLamar1", "KendrickLamar2", "KendrickLamar3",
"LadyGaga1", "LadyGaga2", "LadyGaga3",
"LanaDelRey1", "LanaDelRey2", "LanaDelRey3",
"LilNasX1", "LilNasX2", "LilNasX3",
"LilWayne1", "LilWayne2", "LilWayne3",
"LukeCombs1", "LukeCombs2", "LukeCombs3",
"MileyCyrus1", "MileyCyrus2", "MileyCyrus3",
"MorganWallen1", "MorganWallen2", "MorganWallen3",
"NickiMinaj1", "NickiMinaj2", "NickiMinaj3",
"NoahKahan1", "NoahKahan2", "NoahKahan3",
"OliviaRodrigo1", "OliviaRodrigo2", "OliviaRodrigo3",
"PlayboiCarti1", "PlayboiCarti2", "PlayboiCarti3",
"PostMalone1", "PostMalone2", "PostMalone3",
"Rihanna1", "Rihanna2", "Rihanna3",
"SabrinaCarpenter1", "SabrinaCarpenter2", "SabrinaCarpenter3",
"SelenaGomez1", "SelenaGomez2", "SelenaGomez3",
"Shakira1", "Shakira2", "Shakira3",
"ShawnMendes1", "ShawnMendes2", "ShawnMendes3",
"SZA1", "SZA2", "SZA3",
"TateMcRae1", "TateMcRae2", "TateMcRae3",
"TaylorSwift1", "TaylorSwift2", "TaylorSwift3",
"TheWeeknd1", "TheWeeknd2", "TheWeeknd3",
"TravisScott1", "TravisScott2", "TravisScott3",
"TylerTheCreator1", "TylerTheCreator2", "TylerTheCreator3",
"ZachBryan1", "ZachBryan2", "ZachBryan3"
]

speaker_wavs = [
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Adele/Adele - Hello (Official Music Video).mp3", # Adele1
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Adele/Adele - Rolling in the Deep (Official Music Video).mp3", # Adele2
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Adele/Adele - Someone Like You (Official Music Video).mp3", # Adele3
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ArianaGrande/Ariana Grande - no tears left to cry (Official Video).mp3", # ArianaGrande1
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ArianaGrande/Ariana Grande - Dangerous Woman (Official Video).mp3", # ArianaGrande2
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ArianaGrande/Ariana Grande - thank u, next (Official Video).mp3", # ArianaGrande3
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/BadBunny/BAD BUNNY - BAILE INoLVIDABLE (Video Oficial) DeBÍ TiRAR MáS FOToS.mp3", # BadBunny1
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/BadBunny/Bad Bunny - DtMF (Letra).mp3", # BadBunny2
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/BadBunny/Bad Bunny - Tití Me Preguntó (Video Oficial) Un Verano Sin Ti.mp3", # BadBunny3
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Beyonce/Beyoncé - 711.mp3", # Beyonce1
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Beyonce/Beyoncé - Halo.mp3", # Beyonce2
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Beyonce/Beyoncé - Single Ladies (Put a Ring on It) (Video Version).mp3", # Beyonce3
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/BillieElilish/Billie Eilish - BIRDS OF A FEATHER (Official Music Video).mp3", # BillieEilish1
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/BillieElilish/Billie Eilish - What Was I Made For_ (Official Music Video).mp3", # BillieEilish2
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/BillieElilish/Billie Eilish - WILDFLOWER (Official Lyric Video).mp3", # BillieEilish3
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/BrunoMars/Bruno Mars - Just The Way You Are (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/BrunoMars/Bruno Mars - Thats What I Like [Official Music Video].mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/BrunoMars/Bruno Mars - The Lazy Song (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/CamilaCabello/Camila Cabello - Don't Go Yet (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/CamilaCabello/Camila Cabello - Never Be the Same.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/CamilaCabello/Camila Cabello - Shameless (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/CardiB/Cardi B - Be Careful [Official Video].mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/CardiB/Cardi B - Bodak Yellow [OFFICIAL MUSIC VIDEO].mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/CardiB/Cardi B - Thru Your Phone [Official Audio].mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ChappellRoan/Chappell Roan - Good Luck, Babe! (Official Lyric Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ChappellRoan/Chappell Roan - HOT TO GO! (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ChappellRoan/Chappell Roan - Pink Pony Club (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ChrisBrown/Chris Brown - Residuals (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ChrisBrown/Chris Brown - Under The Influence (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ChrisBrown/Chris Brown - Yo (Excuse Me Miss) (Official HD Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/DemiLovato/Demi Lovato - Cool for the Summer (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/DemiLovato/Demi Lovato - Heart Attack (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/DemiLovato/Demi Lovato - Sorry Not Sorry (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/DojaCat/Doja Cat - Need to Know (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/DojaCat/Doja Cat - Paint The Town Red (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/DojaCat/Doja Cat - Streets (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Drake/Drake - Energy.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Drake/Drake - God's Plan.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Drake/Drake - Hotline Bling.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/DuaLipa/Dua Lipa - Houdini (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/DuaLipa/Dua Lipa - Love Again (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/DuaLipa/Dua Lipa - New Rules (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/EdSheeran/Ed Sheeran - Perfect (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/EdSheeran/Ed Sheeran - Shape of You (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/EdSheeran/Ed Sheeran - Thinking Out Loud (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Eminem/Eminem - Lose Yourself.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Eminem/Eminem - Rap God (Explicit).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Eminem/Eminem - The Real Slim Shady (Official Video - Clean Version).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/FrankOcean/Frank Ocean - Ivy.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/FrankOcean/Frank Ocean - Nights.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/FrankOcean/Frank Ocean - Pink White.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/GracieAbrams/Gracie Abrams - I Love You, Im Sorry (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/GracieAbrams/Gracie Abrams - Risk (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/GracieAbrams/Gracie Abrams - Thats So True (Lyrics).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/HarryStyles/Harry Styles - Adore You (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/HarryStyles/Harry Styles - As It Was (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/HarryStyles/Harry Styles - Sign of the Times (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Hozier/Hozier - Take Me To Church.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Hozier/Hozier - Too Sweet (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Hozier/Hozier - Work Song (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/JayZ/JAY-Z - Dirt Off Your Shoulder.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/JayZ/JAY-Z - Song Cry.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/JayZ/JAY-Z - The Story of O.J..mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/JCole/J. Cole - MIDDLE CHILD (Official Audio).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/JCole/J. Cole - Wet Dreamz (1).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/JCole/No Role Modelz.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/JustinBieber/Justin Bieber - Come Around Me (Audio).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/JustinBieber/Justin Bieber - Ghost.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/JustinBieber/Justin Bieber - One Time (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/KanyeWest/Kanye West - Can't Tell Me Nothing.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/KanyeWest/Kanye West - Good Morning.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/KanyeWest/Kanye West - Stronger.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/KatyPerry/Katy Perry - Firework (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/KatyPerry/Katy Perry - I Kissed A Girl (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/KatyPerry/Katy Perry - Roar.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/KendrickLamar/Kendrick Lamar - HUMBLE..mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/KendrickLamar/Kendrick Lamar - Not Like Us.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/KendrickLamar/Kendrick Lamar - Swimming Pools (Drank).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LadyGaga/Lady Gaga - Abracadabra (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LadyGaga/Lady Gaga - Bad Romance (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LadyGaga/Lady Gaga - Poker Face (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LanaDelRey/Lana Del Rey - Summertime Sadness (Official Music Video) (1).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LanaDelRey/Lana Del Rey - Summertime Sadness (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LanaDelRey/Lana Del Rey - Young and Beautiful.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LilNasX/Lil Nas X - MONTERO (Call Me By Your Name) (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LilNasX/Lil Nas X - Panini (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LilNasX/Lil Nas X - THATS WHAT I WANT (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LilWayne/Lil Wayne - A Milli.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LilWayne/Lil Wayne - Uproar (Lyrics).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LilWayne/Receipt.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LukeCombs/Luke Combs - Beautiful Crazy (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LukeCombs/Luke Combs - She Got the Best of Me (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/LukeCombs/Luke Combs - Where the Wild Things Are (Official Studio Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/MileyCyrus/Miley Cyrus - Flowers (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/MileyCyrus/Miley Cyrus - Party In The U.S.A. (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/MileyCyrus/Miley Cyrus - We Can't Stop (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/MorganWallen/Morgan Wallen - Chasin' You (Dream Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/MorganWallen/Morgan Wallen - Last Night (One Record At A Time Sessions).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/MorganWallen/Morgan Wallen - Wasted On You (The Dangerous Sessions).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/NickiMinaj/Nicki Minaj - Anaconda.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/NickiMinaj/Nicki Minaj - Super Bass (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/NickiMinaj/Nicki Minaj - Your Love (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/NoahKahan/Noah Kahan - All My Love (Official Lyric Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/NoahKahan/Noah Kahan - Stick Season (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/NoahKahan/Noah Kahan - Youre Gonna Go Far (Official Lyric Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/OliviaRodrigo/Olivia Rodrigo - deja vu (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/OliviaRodrigo/Olivia Rodrigo - drivers license (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/OliviaRodrigo/Olivia Rodrigo - good 4 u (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/PlayboiCarti/Playboi Carti - ILoveUIHateU (Official Audio).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/PlayboiCarti/Playboi Carti - Magnolia (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/PlayboiCarti/Playboi Carti - Sky [Official Video].mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/PostMalone/Post Malone - Better Now (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/PostMalone/Post Malone - Circles (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/PostMalone/Post Malone - White Iverson.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Rihanna/Rihanna - Bitch Better Have My Money (Explicit).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Rihanna/Rihanna - Diamonds.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Rihanna/Rihanna - Don't Stop The Music.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/SabrinaCarpenter/Sabrina Carpenter - Espresso (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/SabrinaCarpenter/Sabrina Carpenter - Please Please Please (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/SabrinaCarpenter/Sabrina Carpenter - Taste (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/SelenaGomez/Selena Gomez - Lose You To Love Me (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/SelenaGomez/Selena Gomez - Same Old Love.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/SelenaGomez/Selena Gomez - Single Soon (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Shakira/Shakira - Soltera (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Shakira/Shakira - Waka Waka (This Time for Africa) (The Official 2010 FIFA World Cup Song).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Shakira/Shakira - Whenever, Wherever (Official HD Video) (1).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ShawnMendes/Shawn Mendes - Mercy (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ShawnMendes/Shawn Mendes - There's Nothing Holdin' Me Back (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ShawnMendes/Shawn Mendes - Treat You Better.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Sza/SZA - Good Days (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Sza/SZA - Kill Bill (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/Sza/SZA - Snooze (Audio).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TateMcrae/Tate McRae - greedy (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TateMcrae/Tate McRae - she's all i wanna be (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TateMcrae/Tate McRae - you broke me first (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TaylorSwift/Taylor Swift - Blank Space.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TaylorSwift/Taylor Swift - Cruel Summer (Official Audio).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TaylorSwift/Taylor Swift - Shake It Off.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TheWeeknd/The Weeknd - Heartless (Official Audio).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TheWeeknd/The Weeknd - Starboy ft. Daft Punk (Official Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TheWeeknd/The Weeknd - The Hills.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TravisScott/Travis Scott - Antidote.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TravisScott/Travis Scott - BUTTERFLY EFFECT.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TravisScott/Travis Scott - HIGHEST IN THE ROOM (Official Music Video).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TylerTheCreator/DOGTOOTH.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TylerTheCreator/SWEET I THOUGHT YOU WANTED TO DANCE (Audio).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/TylerTheCreator/Tyler, The Creator - Are We Still Friends_ (Audio).mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ZachBryan/Zach Bryan - Nine Ball.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ZachBryan/Zach Bryan - Oklahoma Smokeshow.mp3",
"/Users/Tim/ResemblyzerFinal/Resemblyzer/Songs/ZachBryan/Zach Bryan - Something In The Orange.mp3"
]

for i, speaker_wav in enumerate(speaker_wavs):
    speaker_wavs[i] = preprocess_wav(
    speaker_wav
    ) # processes into wav as they did in demo

encoder = VoiceEncoder("cpu")
print("Running the continuous embedding on cpu, this might take a while...")
# processes input file
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=4)

# Compares to our speaker mp3 data
speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
similarity_dict = {
name: cont_embeds @ speaker_embed
for name, speaker_embed in zip(speaker_names, speaker_embeds)
}


def average_top_two_scores_per_artist(input_embed, speaker_names, speaker_embeds):

    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    # Dictionary to collect scores by artist
    artist_scores = defaultdict(list)

    # Loop through all samples
    for name, embed in zip(speaker_names, speaker_embeds):
        score = cosine_similarity(input_embed, embed)
        artist_scores[name].append(score)

    # Average the top 2 scores for each artist
    artist_top2_avg = {}
    for artist, scores in artist_scores.items():
        top_two = sorted(scores, reverse=True)[:2]
        avg_score = sum(top_two) / 2
        artist_top2_avg[artist] = avg_score

    return artist_top2_avg


input_embed = encoder.embed_utterance(wav) # Define input_embed first
artist_score_dict = average_top_two_scores_per_artist(input_embed, speaker_names, speaker_embeds) # Use it here

# Sort and get top artist
top_artist = max(artist_score_dict.items(), key=lambda x: x[1])
print(f"Top match by averaged top-2: {top_artist[0]} with score {top_artist[1]:.4f}")

# dont need interactive demo its just for visualizing
# interactive_diarization(similarity_dict, wav, wav_splits)

# avg similarity for each speaker
speaker_avgs = []
# dict has a set (one of our singers, array of similarity values that were extract from certain points in the audio file)
for name, sim in similarity_dict.items():
    # this takes the avg of all similarities values calculated for the speaker for better comparison results
    speaker_avgs.append((name, sim.mean()))

# sort by highest avg similarity, 0 becomes closest match
speaker_avgs.sort(key=lambda x: x[1], reverse=True)

print()
print("–––––––-")
print(f"file {Path(input_file_path).name}")

# and can print all avgs already sorted array so will be highest match to lowest match
for name, avg in speaker_avgs:
    print(f"Speaker {name} has an average similarity of {avg:.2f}")

print(f"Best match: {speaker_avgs[0][0]} with a score of {speaker_avgs[0][1]}")

# Graph for visualizing the results
# interactive_diarization(similarity_dict, wav, wav_splits)