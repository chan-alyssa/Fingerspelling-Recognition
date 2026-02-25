#extract new ctcword labels that uses transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from readh5 import ReadH5
import argparse
import joblib
import string
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# H5_PATH = "/work/alyssa/workspace/split1.1.h5f86.h5"
# SCALER_PATH = "/work/alyssa/ctcloss/scalermulti384_9a.pkl"
BATCH_SIZE = 16
EPOCHS = 2000
LR = 1e-3
INPUT_DIM = 384
OUTPUT_DIM = 27  # 26 letters + 1 blank for CTC

char_to_idx = {c: i + 1 for i, c in enumerate(string.ascii_lowercase)}  # a=1,...,z=26
idx_to_char = {i: c for c, i in char_to_idx.items()}
idx_to_char[0] = "<BLANK>"

def encode_word(word):
    return [char_to_idx[c] for c in word.lower() if c in char_to_idx]

def group_by_subtitle_and_consecutive_frames(subtitles, frame_numbers, letters=None):
    groups = []
    framewise_letters = [] if letters else None
    current_group = []
    current_letters = []

    prev_subtitle = None
    prev_frame_num = None

    for i, (frame_num, raw_subtitle) in enumerate(zip(frame_numbers, subtitles)):
        # decode subtitle
        subtitle = ""
        if raw_subtitle:
            if isinstance(raw_subtitle, bytes):
                subtitle = raw_subtitle.decode("utf-8").strip()
            else:
                subtitle = str(raw_subtitle).strip()

        if not subtitle:
            prev_subtitle, prev_frame_num = None, None
            continue
        # check if we should start a new group
        if prev_subtitle is None or subtitle != prev_subtitle:# or (prev_frame_num is not None and frame_num != prev_frame_num + 1):
            if current_group:  # save previous group
                groups.append(current_group)
                if letters:
                    framewise_letters.append(current_letters)
            current_group = [i]
            current_letters = [letters[i]] if letters else []
        else:  # same subtitle and consecutive frame, continue group
            current_group.append(i)
            if letters:
                current_letters.append(letters[i])

        prev_subtitle = subtitle
        prev_frame_num = frame_num

    # add last group if not empty
    if current_group:
        groups.append(current_group)
        if letters:
            framewise_letters.append(current_letters)

    return (groups, framewise_letters) if letters else groups

class MultiClassClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, dropout_prob=0.2):
        super(MultiClassClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.layer_norm3 = nn.LayerNorm(hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_prob)

        self.fc4 = nn.Linear(hidden_size // 4, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.layer_norm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x



# class Transpeller(Dataset):
#     def __init__(self, h5_path, scaler):
#         with h5py.File(h5_path, 'r') as f:
#             self.features = f['video/features384'][:]
#             self.subtitles = f['video/subtitle'][:]
#             self.frame_numbers = f['video/frame_number'][:]
#             self.predword = f['video/subtitle']

#         reshaped = self.features.reshape(-1, self.features.shape[-1])
#         self.features = scaler.transform(reshaped).reshape(self.features.shape)

#         self.groups = group_by_subtitle_and_consecutive_frames(self.subtitles, self.frame_numbers)

#     def __len__(self):
#         return len(self.groups)

#     def __getitem__(self, idx):
#         indices = self.groups[idx]
#         x = self.features[indices]

#         raw_label = self.subtitles[indices[0]].decode("utf-8").strip()
#         word = raw_label.split()[0].lower() if raw_label else ""


#         y = encode_word(word)


#         if word == '?':
#             allowed = set()   # means: no restriction
#         else:
#             allowed = set(word)


#         # print(f"Word: '{word}', Frame length: {x.shape[0]}")

#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), allowed  # pass down the allowed chars



class Transpeller(Dataset):
    def __init__(self, h5_path, scaler, scalerlip):
        with h5py.File(h5_path, 'r') as f:
            self.features = f['video/features384'][:]
            self.lipfeatures = f['video/autoavsr'][:]
            self.cleanedlabel = f['video/cleanedlabel'][:]
            self.frame_numbers = f['video/frame_number'][:]
            #self.temporalletter = f['video/temporalletter10'][:]
        reshaped = self.features.reshape(-1, self.features.shape[-1])
        self.feat2 = scaler.transform(reshaped).reshape(self.features.shape)

        reshaped = self.lipfeatures.reshape(-1, self.lipfeatures.shape[-1])
        self.feat1 = scalerlip.transform(reshaped).reshape(self.lipfeatures.shape)
        #self.feature = np.concatenate((feat1, feat2), axis=1)
        # self.augment = augment
        #self.temporal_aug = TemporalAugment() #if augment else None
        self.groups = group_evaluate(self.cleanedlabel, self.frame_numbers)
        print(f"Number of training samples (words): {len(self.groups)}")
    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indices = self.groups[idx]
        x = self.feat2[indices]
        z = self.feat1[indices]
        # print(f"Indices: {indices}")
        # letter = self.temporalletter[indices]
        # print(letter)
        #raw_label = self.subtitles[indices[0]].decode("utf-8").strip()
        # word = raw_label.split()[0].lower() if raw_label else ""

        # y = encode_word(word)
        # # fw_bytes = self.temporalletter[indices]  # shape (num_frames, 1)
        # # # print(fw_bytes.shape)
        # # fw_bytes_flat = fw_bytes.flatten()       # shape (num_frames,)
        # # print(fw_bytes_flat.shape)
        # allowed = set(word)

        # fw_letters_encoded = []

        # for b in fw_bytes_flat:
        #     # Decode safely even if it's numpy or weird type
        #     if isinstance(b, bytes):
        #         ch = b.decode("utf-8", errors="ignore")
        #     else:
        #         ch = str(b)

        #     ch = ch.lower().strip()

        #     # # Map to integer index
            # if ch in char_to_idx:
            #     fw_letters_encoded.append(char_to_idx[ch])
            # else:
            #     fw_letters_encoded.append(0)   # blank for unknown or space
                


        # fw_targets = torch.tensor(fw_letters_encoded, dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float32)
        z =  torch.tensor(z, dtype=torch.float32)
        # Apply augmentation
        # x, z, fw_targets = self.temporal_aug(x, z, fw_targets)
        #y = torch.tensor(y, dtype=torch.long)
        return x, z,



def group_evaluate(cleanedlabel, frame_numbers):
    groups = []
    current_group = []
    prev_frame_num = None


    for i, (label,frame_num) in enumerate(zip(cleanedlabel,frame_numbers)):

        # If same subtitle and frame number consecutive, add to current group
        if label == 0:
          continue
        if label == 1 and prev_frame_num is not None and frame_num - prev_frame_num == 1:
            current_group.append(i)
        # else:
        #     # New group starting
        #     if current_group:
        #         groups.append(current_group)
        #     current_group = [i]

        prev_frame_num = frame_num

    # Add last group if any
    if current_group:
        groups.append(current_group)

    return groups  


class WordCTCDataset(Dataset):
    def __init__(self, h5_path, scaler):
        with h5py.File(h5_path, 'r') as f:
            self.features = f['video/features'][:]
            self.subtitles = f['video/subtitle'][:]
            self.frame_numbers = f['video/frame_number'][:]

        reshaped = self.features.reshape(-1, self.features.shape[-1])
        self.features = scaler.transform(reshaped).reshape(self.features.shape)

        self.groups = group_by_subtitle_and_consecutive_frames(self.subtitles, self.frame_numbers)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indices = self.groups[idx]
        x = self.features[indices]
        raw_label = self.subtitles[indices[0]].decode("utf-8").strip().lower()
        # print(raw_label)

        # Take the first word ONLY
        word = raw_label.split()[0] if raw_label else ""

        # If the word contains "*", make it blank
        if word =="*P":
            word = raw_label.split()[1] if raw_label else ""
        # if word =="*F" or word =="*FS":
        #     word = " "
        # print(raw_label,word)

        # Keep only letters (remove numbers, punctuation, etc)
        word = re.sub(r'[^a-z]', '', word)
        # print(word)
        y = encode_word(word)

        # # --- Priority 1: word inside parentheses ---
        # match = re.search(r'\(([^)]+)\)', raw_label)
        # if match:
        #     word = match.group(1).split()[0]  # take first word inside ()
        # else:
        #     # --- Priority 2: first word if no parentheses ---
        #     parts = raw_label.split()
        #     word = parts[0] if parts else ""

        # # --- Rule 3: remove if word contains '*' ---
        # if "*" in word:
        #     word = ""

        # # --- Optional: remove extra punctuation ---
        # word = re.sub(r'[^a-z]', '', word)  # keep only letters
        # print(word)

        # y = encode_word(word)
        # raw_label = self.subtitles[indices[0]].decode("utf-8").strip()
        # print(raw_label)
        # word = raw_label.split()[0].lower() if raw_label else ""

        # y = encode_word(word)
        # word = raw_label.split()[0].lower() if raw_label else ""

        # dynamically allowed chars for this word

        allowed = set(word)

        # print(x.shape)
        
        return (
            torch.tensor(x, dtype=torch.float32), 
            torch.tensor(y, dtype=torch.long), 
            allowed  # pass down the allowed chars
        )
    
class WordCTCDatasetTest(Dataset):
    def __init__(self, h5_path, scaler):
        with h5py.File(h5_path, 'r') as f:
            self.features = f['video/features384'][:]
            self.subtitles = f['video/subtitle'][:]
            self.frame_numbers = f['video/frame_number'][:]

        reshaped = self.features.reshape(-1, self.features.shape[-1])
        self.features = scaler.transform(reshaped).reshape(self.features.shape)

        self.groups = group_evaluate(self.frame_numbers)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indices = self.groups[idx]
        x = self.features[indices]

        raw_label = self.subtitles[indices[0]].decode("utf-8").strip()
        word = raw_label.split()[0].lower() if raw_label else ""

        y = encode_word(word)

        # print(f"Word: '{word}', Frame length: {x.shape[0]}")

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
def decode_indices(indices):
    decoded = []
    prev = -1
    for i in indices:
        if i != prev and i != 0:  # skip blanks and repeated characters
            decoded.append(idx_to_char[i])
        prev = i
    return ''.join(decoded)

def levenshtein_distance(ref, hyp):
    """Compute the Levenshtein distance between two strings."""
    m, n = len(ref), len(hyp)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]




class TransformerCTC(nn.Module):
    def __init__(self, input_size, num_classes, d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.2):
        super(TransformerCTC, self).__init__()
        self.input_fc = nn.Linear(input_size, d_model)  # project features → d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (B, T, F)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.fc_out = nn.Linear(d_model, num_classes)
        self.fc_out = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(d_model, num_classes)
                                )


    def forward(self, x):
        # x: (B, T, F)
        x = self.input_fc(x)           # (B, T, d_model)
        x = self.pos_encoder(x)        # add position info
        x = self.transformer(x)        # (B, T, d_model)
        x = self.norm(x)
        x = self.fc_out(x)             # (B, T, num_classes)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)




def add_ctcword_per_frame(h5_path, model, scaler, scalerlip, batch_size=16, device='cuda'):
    # Load dataset
    dataset = Transpeller(h5_path,scaler, scalerlip) #WordCTCDataset(h5_path, scaler)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ctc)

    model.eval()
    model.to(device)

    if h5_path.endswith(".h5"):
        read_h5file = ReadH5(h5_path)
    
        f = h5py.File(h5_path, 'a')
        video_grp = f['video']

        # If 'ctcword' exists, delete it
        if 'ctcword' in video_grp:
            del video_grp['ctcword']
            print('deletedmask')
        # if 'predword' in video_grp:
        #     del video_grp['predword']
        #     print('deleted')
        # if 'predletters' in video_grp:
        #     del video_grp['predletters']
        #     print('deleted')

        # Create new 'ctcword' dataset (one string per frame)
        dt = h5py.string_dtype(encoding='utf-8')
        video_grp.create_dataset('ctcword', (video_grp['features384'].shape[0],), dtype=dt)
        #video_grp.create_dataset('predword', (video_grp['features'].shape[0],), dtype=dt)
        # video_grp.create_dataset('predletters', (video_grp['features'].shape[0],), dtype=dt)

        ctcword_ds = video_grp['ctcword']
        # predword_ds = video_grp['predword']
        # predletters_ds = video_grp['predletters']

        frame_idx = 0  # to keep track of frame positions

        with torch.no_grad():
            for inputs, inputslips, input_lengths in loader:
                inputs = inputs.to(device)
                inputslips = inputslips.to(device)
                logits = model(inputs, inputslips)  # (B, T, C)
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)

                # # mask disallowed characters per sample
                # for i, allowed in enumerate(allowed_list):
                #     allowed_idx = [0] + [char_to_idx[c] for c in allowed]  # keep blank
                #     mask = torch.full_like(log_probs[i], float('-inf'))
                #     mask[:, allowed_idx] = 0
                #     log_probs[i] = log_probs[i] + mask
                

                pred_idx = log_probs.argmax(dim=2)  # (B, T)

                for i, length in enumerate(input_lengths):
                    # print(i, length)
                    pred_seq = pred_idx[i][:length].cpu().tolist()
                    decoded_word = decode_indices(pred_seq)
                    # print(target_word, decoded_word)
                    print(f"Frames {frame_idx}–{frame_idx+length-1} | Pred='{decoded_word}'")
# add if cleanedlabel==1 then add

                    # Assign the predicted word to all frames in this group
                    for j in range(length):
                        # print(frame_idx)
                        cleanedlabel= read_h5file.read_sequence_slice('video/cleanedlabel', frame_idx)
                        ctcword_ds[frame_idx] = decoded_word
                        # predword_ds[frame_idx] = target_word
                        # predletters_ds[frame_idx] = 
                        frame_idx += 1

    print(f"CTC words assigned per frame in H5 file: {h5_path}")

def collate_ctc(batch, verbose=False):
    inputs, inputslips = zip(*batch)
    input_lengths = torch.tensor([x.shape[0] for x in inputs])

    max_T = max(input_lengths)
    F = inputs[0].shape[1]
    padded_inputs = torch.zeros(len(inputs), max_T, F)
    padded_lipinputs = torch.zeros(len(inputslips), max_T, inputslips[0].shape[1])
    for i, (x, z) in enumerate(zip(inputs, inputslips)):
        padded_inputs[i, :x.shape[0], :] = x
        padded_lipinputs[i, :z.shape[0], :] = z


    return (
        padded_inputs.to(DEVICE),
        padded_lipinputs.to(DEVICE),
        input_lengths,

    )



class TransformerVer2(nn.Module):
    def __init__(self,
                 lip_size=768,
                 hand_size=384,
                 d_model=128,
                 nhead=2,
                 num_layers=2,
                 num_classes=27,
                 dropout=0.3):
        super().__init__()

        self.lip_size = lip_size
        self.hand_size = hand_size
        self.d_model = d_model

        # ---------------------------
        # 1) Project each modality
        # ---------------------------
        self.lip_proj = nn.Linear(lip_size, d_model)
        self.hand_proj = nn.Linear(hand_size, d_model)

        # Optional LayerNorm per modality
        self.ln_lip = nn.LayerNorm(d_model)
        self.ln_hand = nn.LayerNorm(d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        # self.gate = nn.Sequential(
        #     nn.Linear(d_model*2, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, 1),
        #     nn.Sigmoid()
        # )
        # self.scalar_gate = nn.Sequential(
        #     nn.Linear(d_model*2, 1),
        #     nn.Sigmoid()
        # )
        # self.pos_enc_fused = PositionalEncoding(d_model*2, dropout=dropout)

        # ---------------------------
        # 2) Optional per-modality temporal transformers
        # ---------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.lip_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hand_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---------------------------
        # 3) Final temporal transformer after fusion
        # ---------------------------
        encoder_layer_fused = nn.TransformerEncoderLayer(
            d_model=d_model*2,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.fused_transformer = nn.TransformerEncoder(encoder_layer_fused, num_layers=num_layers)

        # ---------------------------
        # 4) Output
        # ---------------------------
        self.norm_fused = nn.LayerNorm(d_model*2)
        self.fc_out = nn.Sequential(nn.Linear(d_model*2, d_model),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(d_model, num_classes)
                                )
        #self.fc_out = nn.Linear(d_model*2, num_classes)

    def forward(self, hand_x, lip_x):
        # x: (B, T, F)
        # lip_x, hand_x = torch.split(x, [self.lip_size, self.hand_size], dim=2)

        # ---------------------------
        # Project + normalize
        # ---------------------------
        #lip_p = self.ln_lip(self.lip_proj(lip_x.detach()))  # detach lip gradients if needed
        lip_p = self.ln_lip(self.lip_proj(lip_x))  # detach lip gradients if needed
        hand_p = self.ln_hand(self.hand_proj(hand_x))

        # # ---------------------------
        # # Optional per-modality transformer
        # # ---------------------------

        # fused =  torch.concat([lip_p, hand_p], axis=-1)
        lip_p = self.pos_enc(lip_p)
        hand_p = self.pos_enc(hand_p)
        lip_out = self.lip_transformer(lip_p)
        hand_out = self.hand_transformer(hand_p)
        # gate_input = torch.cat([hand_out,lip_out], dim=-1) 
        # g = self.gate(gate_input)   
        # gated_lip  = g * lip_out
        # gated_hand = (1 - g) * hand_out

        gate_input = torch.cat([lip_out, hand_out], dim=-1)
        # g = self.scalar_gate(gate_input)
        # fused = torch.cat([g * hand_out, (1-g) * lip_out], dim=-1)


        #fused = torch.concat([hand_out, lip_out], axis=-1)
        # fused = self.lip_w * lip_out + self.hand_w * hand_out

        # ---------------------------
        # Temporal transformer + norm
        # ---------------------------
        
        fused = self.fused_transformer(gate_input)
        fused = self.norm_fused(fused)

        # ---------------------------
        # Output per frame
        # ---------------------------
        out = self.fc_out(fused)  # (B, T, num_classes)
        return out
    


def main():
    parser = argparse.ArgumentParser(description='Sliding Window IoU Detection')
    # parser.add_argument('--hfile_dir', type=str, help='Path to the directory containing left.npy and right.npy files')
    # parser.add_argument('--vid', type=str, help='Path to the video file')
    # parser.add_argument('--csv_file', type=str, help='Path to the CSV file containing annotations')
    # parser.add_argument('--startframe', type=int, help='Start frame number for analysis (0-indexed)')
    # parser.add_argument('--endframe', type=int, help='End frame number for analysis (inclusive)')
    # parser.add_argument('--threshold', type = float)
    # parser.add_argument('--out_folder', type=str)
    parser.add_argument('--h5file', type=str)
    parser.add_argument('--scalerlip', type=str)
    parser.add_argument('--scalerhands', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    # SCALER_LIP = "scalerautoavsr.pkl"
    # SCALER_PATH = "scalerhands.pkl"
    # H5_PATH = "ammonite1.mp4.h5"

    print(args.scalerhands)

    scaler = joblib.load(args.scalerhands)
    scalerlip = joblib.load(args.scalerlip)
    model = TransformerVer2(
        lip_size=768,
        hand_size=384
    ).to(DEVICE)

    #model.load_state_dict(torch.load("multiclassifier.pth", map_location=DEVICE))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    model.eval()

    # # Loop through all H5 files
    # for h5_file in folder_path.glob("*.h5"):
    #     print(f"Processing {h5_file.name}...")
    #     add_ctcword_per_frame(str(h5_file), model, scaler, scalerlip, batch_size=BATCH_SIZE, device=DEVICE)
    add_ctcword_per_frame(args.h5file,model, scaler, scalerlip, batch_size=BATCH_SIZE, device=DEVICE)

    # scaler = joblib.load(SCALER_PATH)
    # model = TransformerCTC(INPUT_DIM, OUTPUT_DIM).to(DEVICE)
    # model.load_state_dict(torch.load("/work/alyssa/ctcloss/transformeralldataoversample3.3.pth", map_location=DEVICE))

    # add_ctcword_per_frame(H5_PATH, model, scaler, batch_size=BATCH_SIZE, device=DEVICE)
if __name__ == "__main__":
    main()
