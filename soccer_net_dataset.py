from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="path/to/soccernet")

mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"]) # labels
# mySoccerNetDownloader.downloadDataTask(task="spotting-2023", split=["train", "valid", "test", "challenge"]) # highlight labels for 2023 challenge task
# mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["train", "valid", "test"]) # visual features (not needed)

# mySoccerNetDownloader.password = "Password from NDA" # TODO: we need access
# videos:
# mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"]) # 224p resolution
# mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train", "valid", "test"]) # 720p resolution

# load the Labels-v2.json file
with open("Labels-v2.json", "r") as f:
    labels_data = json.load(f)

# Load the transcripts data
transcripts = pd.read_csv("transcripts.csv", names=["row_id", "start_time", "end_time", "text", "game"])

# convert `gameTime` in Labels-v2.json to seconds
def game_time_to_seconds(game_time):
    half, time_str = game_time.split(" - ")
    minutes, seconds = map(int, time_str.split(":"))
    half = int(half)
    return (half - 1) * 45 * 60 + minutes * 60 + seconds

# add a column to the JSON data with `time_in_seconds`
for entry in labels_data:
    entry["time_in_seconds"] = game_time_to_seconds(entry["gameTime"])

# map highlights to transcripts
def map_highlights_to_transcripts(transcripts, labels_data):
    transcripts["label"] = "non-highlights"
    for _, label_entry in enumerate(labels_data):
        time_sec = label_entry["time_in_seconds"]
        match_game = label_entry["team"]  # Assuming game matching is needed
        label = label_entry["label"]
        
        # update label in transcripts
        mask = (
            (transcripts["start_time"] <= time_sec) & 
            (transcripts["end_time"] >= time_sec)
        )
        transcripts.loc[mask, "label"] = label
    return transcripts

# convert times in transcripts to seconds
transcripts["start_time"] = transcripts["start_time"].astype(float)
transcripts["end_time"] = transcripts["end_time"].astype(float)

# apply the mapping
labeled_transcripts = map_highlights_to_transcripts(transcripts, labels_data)

# save to a new DataFrame
final_df = labeled_transcripts[["start_time", "end_time", "game", "text", "label"]]

# output the first few rows to verify
print(final_df.head())

# save the processed DataFrame
final_df.to_csv("labeled_transcripts.csv", index=False)
