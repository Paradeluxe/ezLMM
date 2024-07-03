import pandas as pd
import os


if __name__ == "__main__":
    animal = {'monkey': 2, 'parrot': 2, 'goldfish': 2, 'dolphin': 2, 'turkey': 2, 'spider': 2,
              'turtle': 2, 'penguin': 2, 'seahorse': 2, 'giraffe': 2, 'donkey': 2, 'pigeon': 2,
              'elephant': 3, 'dinosaur': 3, 'kangaroo': 3, 'mosquito': 3, 'jellyfish': 3, 'lionfish': 3,
              'gorilla': 3, 'ladybug': 3, 'lyrebird': 3, 'octopus': 3, 'canary': 3, 'antelope': 3}
    nonanimal = {'flower': 2, 'pencil': 2, 'coffee': 2, 'sunshine': 2, 'mountain': 2, 'onion': 2,
                 'guitar': 2, 'basket': 2, 'violin': 2, 'carrot': 2, 'rubbish': 2, 'software': 2,
                 'potato': 3, 'banana': 3, 'bicycle': 3, 'umbrella': 3, 'tomato': 3, 'energy': 3,
                 'medicine': 3, 'anxiety': 3, 'gravity': 3, 'musical': 3, 'cucumber': 3, 'balcony': 3}
    words = {'monkey': 2, 'parrot': 2, 'goldfish': 2, 'dolphin': 2, 'turkey': 2, 'spider': 2,
              'turtle': 2, 'penguin': 2, 'seahorse': 2, 'giraffe': 2, 'donkey': 2, 'pigeon': 2,
              'elephant': 3, 'dinosaur': 3, 'kangaroo': 3, 'mosquito': 3, 'jellyfish': 3, 'lionfish': 3,
              'gorilla': 3, 'ladybug': 3, 'lyrebird': 3, 'octopus': 3, 'canary': 3, 'antelope': 3,
             'flower': 2, 'pencil': 2, 'coffee': 2, 'sunshine': 2, 'mountain': 2, 'onion': 2,
             'guitar': 2, 'basket': 2, 'violin': 2, 'carrot': 2, 'rubbish': 2, 'software': 2,
             'potato': 3, 'banana': 3, 'bicycle': 3, 'umbrella': 3, 'tomato': 3, 'energy': 3,
             'medicine': 3, 'anxiety': 3, 'gravity': 3, 'musical': 3, 'cucumber': 3, 'balcony': 3}
    path = r"C:\Users\Lenovo-PC\Desktop\data\39-yinziyue_linguistic_rhythm_priming_2023-03-09_16h24.24.308.csv"
    df = pd.read_csv(path)
    new_df = pd.DataFrame()

    # sub	priming	word	ifanimal	ifcorr	rt	score_level	acc	fam	familiarity	comprehensibility	concretness	score	diff	exp_type	syl
    df = df[~pd.isna(df["priming_type"])]  # 去除空行

    df["sub"] = os.path.basename(path)


    df.loc[df['priming_type'].str.contains('primingeq'), 'priming'] = 'primingeq'
    df.loc[~df['priming_type'].str.contains('primingeq'), 'priming'] = 'priming'

    df.loc[df['target_audio'].str.contains('altered'), 'exp_type'] = "exp2"
    df.loc[~df['target_audio'].str.contains('altered'), 'exp_type'] = "exp1"

    df["word"] = df["target_audio"].str.split('_').str.get(1)

    df["ifanimal"] = df["word"].isin(animal)

    df["ifcorr"] = ((df["ifanimal"] == True & (df["key_resp_2.keys"] == "j")) | (df["ifanimal"] == False & (df["key_resp_2.keys"] == "f"))).astype(int)

    df["syl"] = df["word"].apply(lambda x: words[x])
    df.rename(columns={'key_resp_2.rt': 'rt'}, inplace=True)

    # df.rename(columns={'priming_type': 'priming'}, inplace=True)

    print(df)
    df = df[["sub", "priming", "word", "ifanimal", "ifcorr", "rt", "exp_type", "syl"]]

    df.to_csv(r"C:\Users\Lenovo-PC\Desktop\test.csv", index=False)
