"""
Top 50 artists in The United States that we support for vocal identification (All song samples contain no feautures):
-–––––––––––––––––––––
Taylor Swift
Billie Eilish
Kendrick Lamar
The Weeknd
Rihanna
Ariana Grande
Ed Sheeran
Beyoncé
Adele
Justin Bieber
Drake
Lady Gaga
Katy Perry
Bruno Mars
Post Malone
Miley Cyrus
SZA
Shakira
Chris Brown
Kanye West
Lana Del Rey
Demi Lovato
J. Cole
Travis Scott
Gracie Abrams
Lil Nas X
Doja Cat
Harry Styles
Olivia Rodrigo
Dua Lipa
Bad Bunny
Nicki Minaj
Eminem
Jay-Z
Lil Wayne
Cardi B
Selena Gomez
Camila Cabello
Shawn Mendes
Sabrina Carpenter
Morgan Wallen
Playboi Carti
Chappell Roan
Zach Bryan
Hozier
Luke Combs
Frank Ocean
Tyler, the Creator
Tate McRae
Noah Kahan
"""

from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import numpy as np
import time, os, warnings
from concurrent.futures import ThreadPoolExecutor

from collections import defaultdict
from numpy import dot
from numpy.linalg import norm

from shutil import get_terminal_size # uses to get width of terminal

from colorama import init, Fore
init(autoreset=True)  # Initialize colorama


# REFRENCE AUIDO 2 array MUST BE IN MATCHING ORDER
# speaker_names = [speaker1, speaker2, ...]
# speaker_wavs = [speaker1_audio_path.mp3, speaker2_audio_path.mp3, ...]

global speaker_wavs, speaker_names

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
    "Adele/Adele - Hello (Official Music Video).mp3",  # Adele1
    "Adele/Adele - Rolling in the Deep (Official Music Video).mp3",  # Adele2
    "Adele/Adele - Someone Like You (Official Music Video).mp3",  # Adele3
    "ArianaGrande/Ariana Grande - no tears left to cry (Official Video).mp3",  # ArianaGrande1
    "ArianaGrande/Ariana Grande - Dangerous Woman (Official Video).mp3",  # ArianaGrande2
    "ArianaGrande/Ariana Grande - thank u, next (Official Video).mp3",  # ArianaGrande3
    "BadBunny/BAD BUNNY - BAILE INoLVIDABLE (Video Oficial)  DeBÍ TiRAR MáS FOToS.mp3",  # BadBunny1
    "BadBunny/Bad Bunny - DtMF (Letra).mp3",  # BadBunny2
    "BadBunny/Bad Bunny - Tití Me Preguntó (Video Oficial)  Un Verano Sin Ti.mp3",  # BadBunny3
    "Beyonce/Beyoncé - 711.mp3",  # Beyonce1
    "Beyonce/Beyoncé - Halo.mp3",  # Beyonce2
    "Beyonce/Beyoncé - Single Ladies (Put a Ring on It) (Video Version).mp3",  # Beyonce3
    "BillieElilish/Billie Eilish - BIRDS OF A FEATHER (Official Music Video).mp3",  # BillieEilish1
    "BillieElilish/Billie Eilish - What Was I Made For_ (Official Music Video).mp3",  # BillieEilish2
    "BillieElilish/Billie Eilish - WILDFLOWER (Official Lyric Video).mp3",  # BillieEilish3
    "BrunoMars/Bruno Mars - Just The Way You Are (Official Music Video).mp3",
    "BrunoMars/Bruno Mars - Thats What I Like [Official Music Video].mp3",
    "BrunoMars/Bruno Mars - The Lazy Song (Official Music Video).mp3",
    "CamilaCabello/Camila Cabello - Don't Go Yet (Official Video).mp3",
    "CamilaCabello/Camila Cabello - Never Be the Same.mp3",
    "CamilaCabello/Camila Cabello - Shameless (Official Video).mp3",
    "CardiB/Cardi B - Be Careful [Official Video].mp3",
    "CardiB/Cardi B - Bodak Yellow [OFFICIAL MUSIC VIDEO].mp3",
    "CardiB/Cardi B - Thru Your Phone [Official Audio].mp3",
    "ChappellRoan/Chappell Roan - Good Luck, Babe! (Official Lyric Video).mp3",
    "ChappellRoan/Chappell Roan - HOT TO GO! (Official Music Video).mp3",
    "ChappellRoan/Chappell Roan - Pink Pony Club (Official Music Video).mp3",
    "ChrisBrown/Chris Brown - Residuals (Official Video).mp3",
    "ChrisBrown/Chris Brown - Under The Influence (Official Video).mp3",
    "ChrisBrown/Chris Brown - Yo (Excuse Me Miss) (Official HD Video).mp3",
    "DemiLovato/Demi Lovato - Cool for the Summer (Official Video).mp3",
    "DemiLovato/Demi Lovato - Heart Attack (Official Video).mp3",
    "DemiLovato/Demi Lovato - Sorry Not Sorry (Official Video).mp3",
    "DojaCat/Doja Cat - Need to Know (Official Video).mp3",
    "DojaCat/Doja Cat - Paint The Town Red (Official Video).mp3",
    "DojaCat/Doja Cat - Streets (Official Video).mp3",
    "Drake/Drake - Energy.mp3",
    "Drake/Drake - God's Plan.mp3",
    "Drake/Drake - Hotline Bling.mp3",
    "DuaLipa/Dua Lipa - Houdini (Official Music Video).mp3",
    "DuaLipa/Dua Lipa - Love Again (Official Music Video).mp3",
    "DuaLipa/Dua Lipa - New Rules (Official Music Video).mp3",
    "EdSheeran/Ed Sheeran - Perfect (Official Music Video).mp3",
    "EdSheeran/Ed Sheeran - Shape of You (Official Music Video).mp3",
    "EdSheeran/Ed Sheeran - Thinking Out Loud (Official Music Video).mp3",
    "Eminem/Eminem - Lose Yourself.mp3",
    "Eminem/Eminem - Rap God (Explicit).mp3",
    "Eminem/Eminem - The Real Slim Shady (Official Video - Clean Version).mp3",
    "FrankOcean/Frank Ocean - Ivy.mp3",
    "FrankOcean/Frank Ocean - Nights.mp3",
    "FrankOcean/Frank Ocean - Pink  White.mp3",
    "GracieAbrams/Gracie Abrams - I Love You, Im Sorry (Official Music Video).mp3",
    "GracieAbrams/Gracie Abrams - Risk (Official Music Video).mp3",
    "GracieAbrams/Gracie Abrams - Thats So True (Lyrics).mp3",
    "HarryStyles/Harry Styles - Adore You (Official Video).mp3",
    "HarryStyles/Harry Styles - As It Was (Official Video).mp3",
    "HarryStyles/Harry Styles - Sign of the Times (Official Video).mp3",
    "Hozier/Hozier - Take Me To Church.mp3",
    "Hozier/Hozier - Too Sweet (Official Video).mp3",
    "Hozier/Hozier - Work Song (Official Video).mp3",
    "JayZ/JAY-Z - Dirt Off Your Shoulder.mp3",
    "JayZ/JAY-Z - Song Cry.mp3",
    "JayZ/JAY-Z - The Story of O.J..mp3",
    "JCole/J. Cole - MIDDLE CHILD (Official Audio).mp3",
    "JCole/J. Cole - Wet Dreamz (1).mp3",
    "JCole/No Role Modelz.mp3",
    "JustinBieber/Justin Bieber - Come Around Me (Audio).mp3",
    "JustinBieber/Justin Bieber - Ghost.mp3",
    "JustinBieber/Justin Bieber - One Time (Official Music Video).mp3",
    "KanyeWest/Kanye West - Can't Tell Me Nothing.mp3",
    "KanyeWest/Kanye West - Good Morning.mp3",
    "KanyeWest/Kanye West - Stronger.mp3",
    "KatyPerry/Katy Perry - Firework (Official Music Video).mp3",
    "KatyPerry/Katy Perry - I Kissed A Girl (Official Music Video).mp3",
    "KatyPerry/Katy Perry - Roar.mp3",
    "KendrickLamar/Kendrick Lamar - HUMBLE..mp3",
    "KendrickLamar/Kendrick Lamar - Not Like Us.mp3",
    "KendrickLamar/Kendrick Lamar - Swimming Pools (Drank).mp3",
    "LadyGaga/Lady Gaga - Abracadabra (Official Music Video).mp3",
    "LadyGaga/Lady Gaga - Bad Romance (Official Music Video).mp3",
    "LadyGaga/Lady Gaga - Poker Face (Official Music Video).mp3",
    "LanaDelRey/Lana Del Rey - Summertime Sadness (Official Music Video) (1).mp3",
    "LanaDelRey/Lana Del Rey - Summertime Sadness (Official Music Video).mp3",
    "LanaDelRey/Lana Del Rey - Young and Beautiful.mp3",
    "LilNasX/Lil Nas X - MONTERO (Call Me By Your Name) (Official Video).mp3",
    "LilNasX/Lil Nas X - Panini (Official Video).mp3",
    "LilNasX/Lil Nas X - THATS WHAT I WANT (Official Video).mp3",
    "LilWayne/Lil Wayne - A Milli.mp3",
    "LilWayne/Lil Wayne - Uproar (Lyrics).mp3",
    "LilWayne/Receipt.mp3",
    "LukeCombs/Luke Combs - Beautiful Crazy (Official Video).mp3",
    "LukeCombs/Luke Combs - She Got the Best of Me (Official Video).mp3",
    "LukeCombs/Luke Combs - Where the Wild Things Are (Official Studio Video).mp3",
    "MileyCyrus/Miley Cyrus - Flowers (Official Video).mp3",
    "MileyCyrus/Miley Cyrus - Party In The U.S.A. (Official Video).mp3",
    "MileyCyrus/Miley Cyrus - We Can't Stop (Official Video).mp3",
    "MorganWallen/Morgan Wallen - Chasin' You (Dream Video).mp3",
    "MorganWallen/Morgan Wallen - Last Night (One Record At A Time Sessions).mp3",
    "MorganWallen/Morgan Wallen - Wasted On You (The Dangerous Sessions).mp3",
    "NickiMinaj/Nicki Minaj - Anaconda.mp3",
    "NickiMinaj/Nicki Minaj - Super Bass (Official Video).mp3",
    "NickiMinaj/Nicki Minaj - Your Love (Official Video).mp3",
    "NoahKahan/Noah Kahan - All My Love (Official Lyric Video).mp3",
    "NoahKahan/Noah Kahan - Stick Season (Official Music Video).mp3",
    "NoahKahan/Noah Kahan - Youre Gonna Go Far (Official Lyric Video).mp3",
    "OliviaRodrigo/Olivia Rodrigo - deja vu (Official Video).mp3",
    "OliviaRodrigo/Olivia Rodrigo - drivers license (Official Video).mp3",
    "OliviaRodrigo/Olivia Rodrigo - good 4 u (Official Video).mp3",
    "PlayboiCarti/Playboi Carti - ILoveUIHateU (Official Audio).mp3",
    "PlayboiCarti/Playboi Carti - Magnolia (Official Video).mp3",
    "PlayboiCarti/Playboi Carti - Sky [Official Video].mp3",
    "PostMalone/Post Malone - Better Now (Official Video).mp3",
    "PostMalone/Post Malone - Circles (Official Music Video).mp3",
    "PostMalone/Post Malone - White Iverson.mp3",
    "Rihanna/Rihanna - Bitch Better Have My Money (Explicit).mp3",
    "Rihanna/Rihanna - Diamonds.mp3",
    "Rihanna/Rihanna - Don't Stop The Music.mp3",
    "SabrinaCarpenter/Sabrina Carpenter - Espresso (Official Video).mp3",
    "SabrinaCarpenter/Sabrina Carpenter - Please Please Please (Official Video).mp3",
    "SabrinaCarpenter/Sabrina Carpenter - Taste (Official Video).mp3",
    "SelenaGomez/Selena Gomez - Lose You To Love Me (Official Music Video).mp3",
    "SelenaGomez/Selena Gomez - Same Old Love.mp3",
    "SelenaGomez/Selena Gomez - Single Soon (Official Music Video).mp3",
    "Shakira/Shakira - Soltera (Official Video).mp3",
    "Shakira/Shakira - Waka Waka (This Time for Africa) (The Official 2010 FIFA World Cup Song).mp3",
    "Shakira/Shakira - Whenever, Wherever (Official HD Video) (1).mp3",
    "ShawnMendes/Shawn Mendes - Mercy (Official Music Video).mp3",
    "ShawnMendes/Shawn Mendes - There's Nothing Holdin' Me Back (Official Music Video).mp3",
    "ShawnMendes/Shawn Mendes - Treat You Better.mp3",
    "Sza/SZA - Good Days (Official Video).mp3",
    "Sza/SZA - Kill Bill (Official Video).mp3",
    "Sza/SZA - Snooze (Audio).mp3",
    "TateMcrae/Tate McRae - greedy (Official Video).mp3",
    "TateMcrae/Tate McRae - she's all i wanna be (Official Video).mp3",
    "TateMcrae/Tate McRae - you broke me first (Official Video).mp3",
    "TaylorSwift/Taylor Swift - Blank Space.mp3",
    "TaylorSwift/Taylor Swift - Cruel Summer (Official Audio).mp3",
    "TaylorSwift/Taylor Swift - Shake It Off.mp3",
    "TheWeeknd/The Weeknd - Heartless (Official Audio).mp3",
    "TheWeeknd/The Weeknd - Starboy ft. Daft Punk (Official Video).mp3",
    "TheWeeknd/The Weeknd - The Hills.mp3",
    "TravisScott/Travis Scott - Antidote.mp3",
    "TravisScott/Travis Scott - BUTTERFLY EFFECT.mp3",
    "TravisScott/Travis Scott - HIGHEST IN THE ROOM (Official Music Video).mp3",
    "TylerTheCreator/DOGTOOTH.mp3",
    "TylerTheCreator/SWEET  I THOUGHT YOU WANTED TO DANCE (Audio).mp3",
    "TylerTheCreator/Tyler, The Creator - Are We Still Friends_ (Audio).mp3",
    "ZachBryan/Zach Bryan - Nine Ball.mp3",
    "ZachBryan/Zach Bryan - Oklahoma Smokeshow.mp3",
    "ZachBryan/Zach Bryan - Something In The Orange.mp3"
]


refrence_folder_path = "idenification_refrences/"
np_save_dir = "processed_refrences/"

# combine base folder path with all audio file paths
for i, speaker_wav_path in enumerate(speaker_wavs):
    speaker_wavs[i] = os.path.join(refrence_folder_path, speaker_wav_path)



warnings.filterwarnings("ignore")

def print_carriage_return(message):
    shell_width = get_terminal_size().columns
    
    print("\r" + " " * shell_width, end="\r", flush=True)
    
    print("\r" + message, end="\r", flush=True)


def trim_wav(wav, length=20):
    return wav[length * 16000:-20 * length]  # assuming 16kHz sample rate

def save_waveform_data(wav: np.ndarray, save_path):
    """
    Saves the waveform data to a .npy file.

    :param wav: The waveform as a numpy array.
    :param save_path: The path where the waveform data will be saved.
    """

    # create directories if not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(str(save_path), wav)

def load_waveform_data(file_path):
    """
    Loads the waveform data from a .npy file.

    :param file_path: The path to the .npy file.
    :return: The loaded waveform as a numpy array.
    """
    return np.load(str(file_path), allow_pickle=True)

def og_path_to_save_path(og_path : str):
    return og_path.replace(".mp3", ".npy").replace(" ", "").strip().replace("-", "_")

def threaded_load(wavs, workers=3):
    processed_wavs = [None] * len(wavs)
    
    curr = 1
    total = len(wavs)

    def worker(index, path, processed_wavs):
        nonlocal curr, total
        save_path = og_path_to_save_path(path)
        save_path = save_path.replace(refrence_folder_path, np_save_dir)
        wav = None
        if os.path.exists(save_path):
            wav = load_waveform_data(save_path)  # loads from disk
        else:
            wav = preprocess_wav(
                path
            )  # processes into wav
            wav = trim_wav(wav)
            save_waveform_data(wav, save_path)  # saves to disk
        processed_wavs[index] = wav  # update the original array with the processed wav
        print_carriage_return(f"{Fore.CYAN}Loaded Ref {curr}/{total} ({int(curr/total*100)}%)")
        curr += 1


    print(f"{Fore.GREEN}Loading audio | Max workers={workers}")

    # load audio with multiple workers
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, i, path, processed_wavs) for i, path in enumerate(wavs)]

    # Wait for all futures to complete
    for future in futures:
        future.result()  # This will also raise exceptions if any occurred in the worker

    return processed_wavs

def threaded_identify(encoder, wavs, workers=3):
    # Initialize an empty list with len(wavs) for speaker embeddings
    speaker_embeds = [None] * len(wavs)
    curr = 1
    total = len(speaker_embeds)
    def worker(index, sp_wav):
        nonlocal curr, total
        speaker_embed = encoder.embed_utterance(sp_wav)
        speaker_embeds[index] = speaker_embed

        # print curr progress (\r)
        print_carriage_return(f"{Fore.CYAN}Compared to Ref {curr}/{total} ({int(curr/total*100)}%)")
        curr+=1

    print(f"{Fore.GREEN}Starting threads | Max workers={workers}")
    # Identify with multiple workers
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, i, wav) for i, wav in enumerate(wavs)]

    # Wait for all futures to complete
    for future in futures:
        future.result()

    return speaker_embeds

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

def sorted_artist_averages(sim_dict):
    # avg similarity for each speaker
    speaker_avgs = {}

    # each speaker has <name>(1|2|3), avg the 3 from each speaker
    for name, sim in sim_dict.items():
        name = name[:-1] # remove the 1|2|3
        if name not in speaker_avgs:
            speaker_avgs[name] = [sim.mean()]
            # speaker_avgs[name] = sim.mean() / 3
        else:
            speaker_avgs[name].append(sim.mean())
            # speaker_avgs[name] += sim.mean() / 3
    for name in speaker_avgs.keys():
        # avg the 3 samples for each speaker
        speaker_avgs[name].sort(reverse=True)

        high = 1
        mid = 0
        low = 0
        print(f"({name}): {speaker_avgs[name]}")
        if len(speaker_avgs[name]) == 3:
            speaker_avgs[name] = (speaker_avgs[name][0] * high) + (speaker_avgs[name][1] * mid) + (speaker_avgs[name][2] * low)
        else:
            speaker_avgs[name] = np.mean(speaker_avgs[name])

    # turn speaker avgs into tuple array sorted by mean
    speaker_avgs = [(k, v) for k, v in speaker_avgs.items()]
    speaker_avgs.sort(key=lambda x: x[1], reverse=True)
    return speaker_avgs

# Terminal color
class identifier:
    def __init__(self, device="cpu"):

        self.device = device = "cpu" # this model requires a lot of vram which makes it impratical to run on our gpus even though we run the detection models on gpu
        print("---")
        print(f"Initializing VocieEncoder on {Fore.CYAN}{self.device}")
        # We can load this model into RAM to skip that time step in future
        self.encoder = VoiceEncoder(self.device)
        print("---")


    def identify_file(self, input_file_path, max_workers_load=5, max_workers_identify=10):
        global speaker_wavs, speaker_names
        # process wavs
        # will save processed data to disk if not already saved to improve speed on rerun
        print_carriage_return(f"{Fore.YELLOW}Loading audio...")
        speaker_wavs_loaded = threaded_load(wavs=speaker_wavs,workers=max_workers_load)

        print_carriage_return(f"{Fore.YELLOW}Processing input file...")
        wav_fpath = Path(input_file_path)  # turns string into path cause thats what lib wants
        wav = preprocess_wav(wav_fpath)
        # processes input file
        _, cont_embeds, wav_splits = self.encoder.embed_utterance(wav, return_partials=True, rate=4)

        # Compares to our speaker mp3 data
        print_carriage_return(f"{Fore.YELLOW}Comparing...")
        speaker_embeds = threaded_identify(encoder=self.encoder, wavs=speaker_wavs_loaded, workers=max_workers_identify)

        # Initialize an empty dictionary for similarity
        similarity_dict = {}
        for name, speaker_embed in zip(speaker_names, speaker_embeds):
            similarity_dict[name] = cont_embeds @ speaker_embed
        del speaker_wavs_loaded
        return similarity_dict, speaker_names, speaker_embeds, wav


    def identify_artist(self, input_file_path):
        """
        Returns matched: artist name, similarity score
        """
        # File to analyze for speaker match
        # path to the audio file you want to analyze
        print(f"{Fore.CYAN}Analyzing: {input_file_path}")
        similarity_dict, _speaker_names, speaker_embeds, wav = self.identify_file(input_file_path=input_file_path)
        # artist_score_dict = average_top_two_scores_per_artist(speaker_names=speaker_names, speaker_embeds=speaker_embeds, input_embed=wav)
        sorted_avgs = sorted_artist_averages(similarity_dict)



        # top 3
        for i in range(3):
            artist = sorted_avgs[i]
    
        # Sort and get top artist
        # top_artist = max(artist_score_dict.items(), key=lambda x: x[1])
        # print(f"Top match by averaged top-2: {top_artist[0]} with score {top_artist[1]:.4f}")

        top_artist = sorted_avgs[0]


        return top_artist[0], top_artist[1]

# main func
if __name__ == "__main__":
    # Initialize the identifier
    id = identifier()
    # Identify an artist
    input_file_path = "/Users/christiankilduff/Downloads/audio/isolated-fake/TaylorSwift-Mine(Taylor’sVersion)(PianoVersion)_sep.mp3"
    artist, score = id.identify_artist(input_file_path=input_file_path)
    print(f"Identified artist: {artist} with score: {score}")