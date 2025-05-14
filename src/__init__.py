import torch
import numpy as np
import os
import subprocess
import librosa
from scipy.stats import skew, kurtosis, mode as scipy_mode
from model import MLP  # Your custom model

def load_model(model_path: str, model: torch.nn.Module, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[DEBUG] ✅ Model loaded from {model_path}")
    return model

def convert_to_wav(input_path: str, output_path: str):
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ac', '1', '-ar', '16000', output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def extract_features(audio_path: str):
    print(f"[DEBUG] 🔍 Extracting features from: {audio_path}")
    temp_wav = "temp.wav"
    convert_to_wav(audio_path, temp_wav)

    try:
        y, sr = librosa.load(temp_wav, sr=16000)
        # Compute spectral centroid per frame (Hz → kHz)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0] / 1000.0  # shape (n_frames,)
        # Basic stats on centroid
        meanfreq = np.mean(cent)
        sd       = np.std(cent)
        median   = np.median(cent)
        Q25      = np.percentile(cent, 25)
        Q75      = np.percentile(cent, 75)
        IQR      = Q75 - Q25
        skewness = skew(cent)
        kurt     = kurtosis(cent)
        # Spectral entropy (normalize power spectrum per frame)
        S = np.abs(librosa.stft(y))
        # Compute power spectrogram
        ps = S**2
        # Sum over frequency bins
        den = np.sum(ps, axis=0, keepdims=True)              # shape (1, n_frames)
        # Replace zeros with eps to avoid 0/0
        den[den == 0] = np.finfo(float).eps                  # machine‐epsilon ~2.2e-16
        # Safe normalization
        ps_norm = ps / den

        sp_ent  = -np.sum(ps_norm * np.log(ps_norm + 1e-10), axis=0).mean()
        # Spectral flatness
        sfm = librosa.feature.spectral_flatness(y=y)[0].mean()
         # Mode of centroid (bin into 100 bins) – handle scalar or array result
        bins = np.linspace(cent.min(), cent.max(), 101)
        digitized = np.digitize(cent, bins)
        mode_res = scipy_mode(digitized, axis=None, keepdims=False)
        # mode_res.mode may be a numpy.int64 or a 1-element array
        mode_idx = int(mode_res.mode)  
        # ensure valid bin index before subtracting 1
        if 1 <= mode_idx <= len(bins):
            mode_cent = bins[mode_idx - 1]
        else:
            mode_cent = 0.0
        # Peak frequency (freq with max energy)
        freqs = librosa.fft_frequencies(sr=sr) / 1000.0
        peak_idx = np.argmax(np.sum(S, axis=1))
        peakf    = freqs[peak_idx]
        # Fundamental frequency stats via pyin
        y_harmonic, _ = librosa.effects.hpss(y)
        f0, _, _      = librosa.pyin(y_harmonic, fmin=50, fmax=400)
        f0 = f0[~np.isnan(f0)] / 1000.0
        if len(f0) == 0:
            raise ValueError("No f0 detected")
        meanfun = np.mean(f0)
        minfun  = np.min(f0)
        maxfun  = np.max(f0)
        # Dominant-frequency periods
        periods = 1.0 / (cent[cent>0])
        meandom = np.mean(periods)
        mindom = np.min(periods)
        maxdom = np.max(periods)
        dfrange= maxdom - mindom
        # Modulation index
        modindx = (np.sum(np.abs(np.diff(f0))) / (maxfun - minfun)
                   if maxfun>minfun else 0.0)

        # Assemble 21 features, then drop peakf for model
        all_feats = [
            meanfreq, sd, median, Q25, Q75, IQR, skewness, kurt,
            sp_ent, sfm, mode_cent, meanfreq, peakf,
            meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx
        ]
        # Final vector excludes peakf (index 12)
        features = [v for i,v in enumerate(all_feats) if i != 12]
        print(f"[DEBUG] ✅ Extracted {len(features)} features")
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        features = [0.0] * 20

    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    arr = np.array(features, dtype=np.float32)
    print(f"[DEBUG] Feature array shape: {arr.shape}")
    print(f"[DEBUG] Values: {arr}")
    return arr

def predict_gender_single(model, audio_path: str, device: torch.device):
    feats = extract_features(audio_path)
    if feats.shape[0] != 20:
        print(f"❌ Wrong feature length: {feats.shape[0]}")
        return "unknown"
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    _, pred = torch.max(out, 1)
    return "male" if pred.item()==0 else "female"

def predict_gender_voting(models, audio_path: str, device: torch.device):
    # Single extraction for all
    feats = extract_features(audio_path)
    if feats.shape[0] != 20:
        return "unknown"

    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    votes = []
    for idx, m in enumerate(models):
        m.eval()
        with torch.no_grad():
            out = m(x)
        _, pred = torch.max(out, 1)
        lab = "male" if pred.item()==0 else "female"
        votes.append(lab)
        print(f"[DEBUG] Model {idx+1} → {lab}")
    male_count = votes.count("male")
    female_count = votes.count("female")
    print(f"[DEBUG] Voting: male={male_count}, female={female_count}")
    return "male" if male_count>female_count else "female"

def main(mp3_path: str, model_dir: str):
    if not os.path.isfile(mp3_path):
        print("❌ MP3 not found")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for fold in range(1,6):
        m = MLP(n_classes=2, input_dim=20).to(device)
        path = os.path.join(model_dir, f"best_model_fold{fold}.pth")
        models.append(load_model(path, m, device))
    gender = predict_gender_voting(models, mp3_path, device)
    print(f"✅ Predicted gender: {gender}")

if __name__ == "__main__":
    mp3_file_path = r"C:\\Users\\abder\\Desktop\\aivrec\\voice_gender_recognition\\test data\\hello_man.mp3"
    model_dir = r"C:\\Users\\abder\\Desktop\\aivrec\\voice_gender_recognition\\output"
    main(mp3_file_path, model_dir)
